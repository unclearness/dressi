#include <algorithm>

#include "core/node.h"
#include "pack/substage.h"

namespace dressi {

bool IsInlineConst(const Variable& var) {
    return var && NodeAccess::Node(var)->const_val.has_value();
}

SubStages TrivialPackSubStages(const Functions& all_funcs) {
    SubStages substages;
    for (const auto& func : all_funcs) {
        Variable y = func.getOutputVar();
        if (IsInlineConst(y)) {
            continue;  // constants are inlined into consumer shaders
        }
        DRESSI_CHECK(func.getShaderType() == FRAG,
                     "Only FRAG functions are executable in Milestone 1");

        SubStage ss;
        ss.funcs = {func};
        ss.out_vars = {y};
        ss.gen_vars = {y};
        ss.shader_type = func.getShaderType();
        ss.img_size = y.getImgSize();
        for (const auto& x : func.getInputVars()) {
            if (IsInlineConst(x)) {
                continue;
            }
            if (std::find(ss.slt_vars.begin(), ss.slt_vars.end(), x) ==
                ss.slt_vars.end()) {
                ss.slt_vars.push_back(x);
            }
        }
        substages.push_back(std::move(ss));
    }
    return substages;
}

Stages WrapSubStagesIntoStages(SubStages substages) {
    Stages stages;
    for (auto& ss : substages) {
        Stage stage;
        stage.inp_vars = ss.inp_vars;
        stage.slt_vars = ss.slt_vars;
        stage.out_vars = ss.out_vars;
        stage.shader_type = ss.shader_type;
        stage.img_size = ss.img_size;
        stage.substages = {std::move(ss)};
        stages.push_back(std::move(stage));
    }
    return stages;
}

}  // namespace dressi
