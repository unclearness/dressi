#include "core/node.h"

#include <atomic>
#include <sstream>
#include <unordered_set>

#include "core/infer.h"

namespace dressi {

void ThrowDressiError(const std::string& msg, const char* file, int line) {
    std::ostringstream ss;
    ss << msg << " (" << file << ":" << line << ")";
    throw DressiError(ss.str());
}

uint32_t NumComponents(VType vtype) {
    switch (vtype) {
        case FLOAT: return 1;
        case VEC2: return 2;
        case VEC3: return 3;
        case VEC4: return 4;
        case MAT2: return 4;
        case MAT3: return 9;
        case MAT4: return 16;
        case INT: return 1;
        case IVEC2: return 2;
        case IVEC3: return 3;
        case IVEC4: return 4;
        case UINT: return 1;
    }
    DRESSI_CHECK(false, "Unknown VType");
}

bool IsIntVType(VType vtype) {
    return vtype == INT || vtype == IVEC2 || vtype == IVEC3 ||
           vtype == IVEC4 || vtype == UINT;
}

bool IsMatrixVType(VType vtype) {
    return vtype == MAT2 || vtype == MAT3 || vtype == MAT4;
}

bool IsVectorVType(VType vtype) {
    return vtype == VEC2 || vtype == VEC3 || vtype == VEC4 ||
           vtype == IVEC2 || vtype == IVEC3 || vtype == IVEC4;
}

const char* GlslTypeName(VType vtype) {
    switch (vtype) {
        case FLOAT: return "float";
        case VEC2: return "vec2";
        case VEC3: return "vec3";
        case VEC4: return "vec4";
        case MAT2: return "mat2";
        case MAT3: return "mat3";
        case MAT4: return "mat4";
        case INT: return "int";
        case IVEC2: return "ivec2";
        case IVEC3: return "ivec3";
        case IVEC4: return "ivec4";
        case UINT: return "uint";
    }
    DRESSI_CHECK(false, "Unknown VType");
}

VType VecVTypeOf(uint32_t n_components, bool is_int) {
    switch (n_components) {
        case 1: return is_int ? INT : FLOAT;
        case 2: return is_int ? IVEC2 : VEC2;
        case 3: return is_int ? IVEC3 : VEC3;
        case 4: return is_int ? IVEC4 : VEC4;
    }
    DRESSI_CHECK(false, "Invalid component count");
}

uint64_t NextNodeId() {
    static std::atomic<uint64_t> s_counter{0};
    return s_counter.fetch_add(1);
}

// ----------------------------------- Variable ---------------------------------
Variable::Variable(VType vtype, ImgSize img_size) {
    m_node = std::make_shared<VariableNode>();
    m_node->id = NextNodeId();
    m_node->vtype = vtype;
    m_node->img_size = img_size;
}

uint64_t Variable::id() const {
    DRESSI_CHECK(m_node, "Null Variable");
    return m_node->id;
}

VType Variable::getVType() const {
    DRESSI_CHECK(m_node, "Null Variable");
    return m_node->vtype;
}

void Variable::setVType(VType vtype) {
    DRESSI_CHECK(m_node, "Null Variable");
    m_node->vtype = vtype;
}

ImgSize Variable::getImgSize() const {
    DRESSI_CHECK(m_node, "Null Variable");
    return m_node->img_size;
}

void Variable::setImgSize(ImgSize img_size) {
    DRESSI_CHECK(m_node, "Null Variable");
    m_node->img_size = img_size;
}

Function Variable::getCreator() const {
    DRESSI_CHECK(m_node, "Null Variable");
    return NodeAccess::Wrap(m_node->creator);
}

void Variable::setCreator(const Function& creator) {
    DRESSI_CHECK(m_node, "Null Variable");
    m_node->creator = NodeAccess::Node(creator);
}

std::vector<Function> Variable::getUsers() const {
    DRESSI_CHECK(m_node, "Null Variable");
    std::vector<Function> users;
    for (const auto& weak_user : m_node->users) {
        if (auto user = weak_user.lock()) {
            users.push_back(NodeAccess::Wrap(std::move(user)));
        }
    }
    return users;
}

void Variable::addUser(const Function& user) {
    DRESSI_CHECK(m_node, "Null Variable");
    DRESSI_CHECK(user, "Null Function as user");
    m_node->users.push_back(NodeAccess::Node(user));
}

bool Variable::getRequiresGrad() const {
    DRESSI_CHECK(m_node, "Null Variable");
    return m_node->req_grad;
}

void Variable::setRequiresGrad(bool req_grad) {
    DRESSI_CHECK(m_node, "Null Variable");
    m_node->req_grad = req_grad;
}

namespace {

// Applies `set` to `start` and every variable computed downstream of it
// (through user functions' outputs).
void ForEachDownstream(const std::shared_ptr<VariableNode>& start,
                       const std::function<void(VariableNode&)>& set) {
    std::unordered_set<const VariableNode*> visited;
    std::vector<std::shared_ptr<VariableNode>> stack = {start};
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!node || !visited.insert(node.get()).second) {
            continue;
        }
        set(*node);
        for (const auto& weak_user : node->users) {
            if (auto user = weak_user.lock()) {
                if (auto y = user->y.lock()) {
                    stack.push_back(std::move(y));
                }
            }
        }
    }
}

}  // namespace

void Variable::setRequiresGradRecursively(bool req_grad) {
    DRESSI_CHECK(m_node, "Null Variable");
    ForEachDownstream(m_node,
                      [req_grad](VariableNode& n) { n.req_grad = req_grad; });
}

bool Variable::IsDirty() const {
    DRESSI_CHECK(m_node, "Null Variable");
    return m_node->is_dirty;
}

void Variable::setDirty(bool is_dirty) {
    DRESSI_CHECK(m_node, "Null Variable");
    m_node->is_dirty = is_dirty;
}

void Variable::setDirtyRecursively(bool is_dirty) {
    DRESSI_CHECK(m_node, "Null Variable");
    ForEachDownstream(m_node,
                      [is_dirty](VariableNode& n) { n.is_dirty = is_dirty; });
}

// ----------------------------------- Function ---------------------------------
Function::Function(const std::string& code, ShaderType type,
                   const BwdFunc& bwd_func) {
    m_node = std::make_shared<FunctionNode>();
    m_node->id = NextNodeId();
    m_node->fwd_code = code;
    m_node->shader_type = type;
    m_node->bwd_func = bwd_func;
}

uint64_t Function::id() const {
    DRESSI_CHECK(m_node, "Null Function");
    return m_node->id;
}

std::string Function::getFwdCode() const {
    DRESSI_CHECK(m_node, "Null Function");
    return m_node->fwd_code;
}

ShaderType Function::getShaderType() const {
    DRESSI_CHECK(m_node, "Null Function");
    return m_node->shader_type;
}

Variables Function::getInputVars() const {
    DRESSI_CHECK(m_node, "Null Function");
    Variables xs;
    xs.reserve(m_node->xs.size());
    for (const auto& x : m_node->xs) {
        xs.push_back(NodeAccess::Wrap(x));
    }
    return xs;
}

Variable Function::getOutputVar() const {
    DRESSI_CHECK(m_node, "Null Function");
    auto y = m_node->y.lock();
    DRESSI_CHECK(y, "Output Variable of Function has expired");
    return NodeAccess::Wrap(std::move(y));
}

Variable Function::buildFwd(const Variables& xs) {
    DRESSI_CHECK(m_node, "Null Function");
    m_node->xs.clear();
    m_node->xs.reserve(xs.size());
    for (const auto& x : xs) {
        DRESSI_CHECK(x, "Null Variable as Function input");
        m_node->xs.push_back(NodeAccess::Node(x));
        Variable x_handle = x;  // handles share the node
        x_handle.addUser(NodeAccess::Wrap(m_node));
    }

    auto y_node = std::make_shared<VariableNode>();
    y_node->id = NextNodeId();
    const auto [vtype, img_size] = InferOutput(*m_node, xs);
    y_node->vtype = vtype;
    y_node->img_size = img_size;
    y_node->creator = m_node;
    m_node->y = y_node;
    return NodeAccess::Wrap(std::move(y_node));
}

Variable Function::buildBwd(const Variable& gy, uint32_t bwd_idx) const {
    DRESSI_CHECK(m_node, "Null Function");
    if (!m_node->bwd_func) {
        return nullptr;
    }
    return m_node->bwd_func(getInputVars(), getOutputVar(), gy, bwd_idx);
}

}  // namespace dressi
