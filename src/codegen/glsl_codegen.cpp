#include "codegen/glsl_codegen.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <sstream>

#include "core/node.h"

namespace dressi {

namespace {

std::string FloatLit(float fv) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", double(fv));
    std::string s(buf);
    if (s.find_first_of(".eEnif") == std::string::npos) {
        s += ".0";
    }
    return s;
}

// GLSL expression for an inline constant variable
std::string ConstExpr(const Variable& var) {
    const auto& node = NodeAccess::Node(var);
    const auto& vals = *node->const_val;
    if (vals.size() == 1) {
        return FloatLit(vals[0]);
    }
    std::string s = std::string(GlslTypeName(var.getVType())) + "(";
    for (size_t i = 0; i < vals.size(); i++) {
        s += (i ? "," : "") + FloatLit(vals[i]);
    }
    return s + ")";
}

const char* SwizzleOf(uint32_t n_comps) {
    switch (n_comps) {
        case 1: return ".x";
        case 2: return ".xy";
        case 3: return ".xyz";
        default: return "";
    }
}

std::string ReplaceAll(std::string s, const std::string& from,
                       const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

}  // namespace

uint32_t PhysChannels(VType vtype) {
    const uint32_t n = NumComponents(vtype);
    DRESSI_CHECK(n <= 4, "Matrix images are not supported on GPU yet (M1)");
    return n == 3 ? 4 : n;
}

std::string FullscreenVertShader() {
    return R"(#version 450
void main() {
    vec2 pos = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
)";
}

std::string GenerateFragShader(const SubStage& ss) {
    DRESSI_CHECK(ss.shader_type == FRAG, "GenerateFragShader: FRAG only");
    DRESSI_CHECK(ss.tex_vars.empty() && ss.uif_vars.empty() &&
                         ss.vtx_vars.empty(),
                 "M1 codegen supports inp/slt inputs only");

    // Substage-local normalized names: inputs first, then generated vars,
    // deterministically ordered for stable golden tests and shader caching.
    std::map<Variable, std::string> names;
    uint32_t next_local = 0;
    const auto local_name = [&](const Variable& v) {
        auto it = names.find(v);
        if (it != names.end()) {
            return it->second;
        }
        const std::string n = "v" + std::to_string(next_local++);
        names.emplace(v, n);
        return n;
    };

    std::ostringstream head;
    std::ostringstream body;
    head << "#version 450\n";

    // Binding convention shared with the executor: inp first, then slt
    for (size_t i = 0; i < ss.inp_vars.size(); i++) {
        const Variable& v = ss.inp_vars[i];
        DRESSI_CHECK(!IsIntVType(v.getVType()),
                     "Int images are not supported on GPU yet (M1)");
        head << "layout(input_attachment_index=" << i << ", set=0, binding="
             << i << ") uniform subpassInput u_inp" << i << ";\n";
    }
    const size_t slt_binding_ofs = ss.inp_vars.size();

    // slt inputs: combined image samplers read via texelFetch
    for (size_t i = 0; i < ss.slt_vars.size(); i++) {
        const Variable& v = ss.slt_vars[i];
        DRESSI_CHECK(!IsIntVType(v.getVType()),
                     "Int images are not supported on GPU yet (M1)");
        head << "layout(set=0, binding=" << (slt_binding_ofs + i)
             << ") uniform sampler2D u_slt" << i << ";\n";
    }

    // Outputs: padded float attachments
    for (size_t i = 0; i < ss.out_vars.size(); i++) {
        const uint32_t phys = PhysChannels(ss.out_vars[i].getVType());
        DRESSI_CHECK(ss.out_vars[i].getImgSize() == ss.img_size,
                     "Substage outputs must share the image size");
        head << "layout(location=" << i << ") out "
             << (phys == 1 ? "float" : ("vec" + std::to_string(phys)))
             << " o_" << i << ";\n";
    }

    body << "void main() {\n";
    body << "    ivec2 dressi_coord = ivec2(gl_FragCoord.xy);\n";

    // Which slt inputs are consumed by coordinate-generic loads (special ops
    // fetch their input themselves)
    const auto is_special = [](const Function& f) {
        const std::string& code = NodeAccess::Node(f)->fwd_code;
        return code == "__reduce_2x2_sum__" ||
               code == "__upsample_nearest_2x__";
    };
    std::map<Variable, size_t> slt_binding;
    for (size_t i = 0; i < ss.slt_vars.size(); i++) {
        slt_binding[ss.slt_vars[i]] = i;
    }
    std::map<Variable, bool> needs_generic_load;
    for (const auto& f : ss.funcs) {
        const bool special = is_special(f);
        for (const auto& x : f.getInputVars()) {
            if (slt_binding.count(x) && !special) {
                needs_generic_load[x] = true;
            }
        }
    }

    // Loads for input attachments (same-pixel, same-size by construction)
    for (size_t i = 0; i < ss.inp_vars.size(); i++) {
        const Variable& v = ss.inp_vars[i];
        const uint32_t n = NumComponents(v.getVType());
        body << "    " << GlslTypeName(v.getVType()) << " " << local_name(v)
             << " = subpassLoad(u_inp" << i << ")" << SwizzleOf(n) << ";\n";
    }

    // Generic loads for slt inputs
    for (size_t i = 0; i < ss.slt_vars.size(); i++) {
        const Variable& v = ss.slt_vars[i];
        if (!needs_generic_load.count(v)) {
            continue;
        }
        const uint32_t n = NumComponents(v.getVType());
        const std::string coord = v.getImgSize().isUniform()
                                          ? "ivec2(0,0)"
                                          : (v.getImgSize() == ss.img_size
                                                     ? "dressi_coord"
                                                     : "");
        DRESSI_CHECK(!coord.empty(),
                     "Non-matching input size requires a special op");
        body << "    " << GlslTypeName(v.getVType()) << " " << local_name(v)
             << " = texelFetch(u_slt" << i << ", " << coord << ", 0)"
             << SwizzleOf(n) << ";\n";
    }

    // Expression for a function input: local name, or inline constant
    const auto input_expr = [&](const Variable& x) -> std::string {
        if (IsInlineConst(x)) {
            return ConstExpr(x);
        }
        auto it = names.find(x);
        DRESSI_CHECK(it != names.end(),
                     "Function input is neither loaded nor generated");
        return it->second;
    };

    // Function bodies in topological (creation id) order
    Functions funcs = ss.funcs;
    std::sort(funcs.begin(), funcs.end(),
              [](const Function& a, const Function& b) {
                  return a.id() < b.id();
              });
    for (const auto& f : funcs) {
        const auto& node = NodeAccess::Node(f);
        const Variable y = f.getOutputVar();
        const Variables xs = f.getInputVars();
        const std::string y_name = local_name(y);
        const char* y_type = GlslTypeName(y.getVType());

        if (node->fwd_code == "__reduce_2x2_sum__") {
            const size_t bind = slt_binding.at(xs[0]);
            const ImgSize src = xs[0].getImgSize();
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name << " = " << y_type
                 << "(0.0);\n";
            body << "    for (int dy = 0; dy < 2; dy++)"
                 << " for (int dx = 0; dx < 2; dx++) {\n";
            body << "        ivec2 p = dressi_coord * 2 + ivec2(dx, dy);\n";
            body << "        if (p.x < " << src.w << " && p.y < " << src.h
                 << ") { " << y_name << " += texelFetch(u_slt" << bind
                 << ", p, 0)" << SwizzleOf(n) << "; }\n";
            body << "    }\n";
            continue;
        }
        if (node->fwd_code == "__upsample_nearest_2x__") {
            const size_t bind = slt_binding.at(xs[0]);
            const uint32_t n = NumComponents(y.getVType());
            body << "    " << y_type << " " << y_name << " = texelFetch(u_slt"
                 << bind << ", dressi_coord / 2, 0)" << SwizzleOf(n) << ";\n";
            continue;
        }

        std::string snippet = node->fwd_code;
        snippet = ReplaceAll(snippet, "{y}", y_name);
        // Component-wise ops (default inference) accept scalar inputs by
        // broadcasting, but GLSL builtins like pow() do not; promote scalar
        // inputs to the output vector type explicitly.
        const bool promote = !node->infer && NumComponents(y.getVType()) > 1;
        for (size_t i = 0; i < xs.size(); i++) {
            std::string expr = input_expr(xs[i]);
            if (promote && NumComponents(xs[i].getVType()) == 1) {
                expr = std::string(y_type) + "(" + expr + ")";
            }
            snippet = ReplaceAll(snippet, "{x" + std::to_string(i) + "}",
                                 expr);
        }
        body << "    " << y_type << " " << y_name << "; " << snippet << "\n";
    }

    // Padded output writes
    for (size_t i = 0; i < ss.out_vars.size(); i++) {
        const Variable& v = ss.out_vars[i];
        const uint32_t n = NumComponents(v.getVType());
        const uint32_t phys = PhysChannels(v.getVType());
        const std::string src = names.at(v);
        if (n == phys) {
            body << "    o_" << i << " = " << src << ";\n";
        } else {  // VEC3 -> vec4 padding
            body << "    o_" << i << " = vec4(" << src << ", 0.0);\n";
        }
    }
    body << "}\n";

    return head.str() + body.str();
}

}  // namespace dressi
