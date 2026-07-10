#include "pack/reactive.h"

#include <algorithm>
#include <unordered_set>

#include "core/node.h"

namespace dressi {

std::set<uint64_t> ComputeDynamicVarIds(const Functions& all_funcs,
                                        const Variables& seeds) {
    std::set<uint64_t> dynamic_ids;
    for (const auto& s : seeds) {
        if (s) {
            dynamic_ids.insert(s.id());
        }
    }
    for (const auto& f : all_funcs) {  // topological order
        bool dyn = false;
        for (const auto& x : f.getInputVars()) {
            if (dynamic_ids.count(x.id())) {
                dyn = true;
                break;
            }
        }
        if (dyn) {
            dynamic_ids.insert(f.getOutputVar().id());
        }
    }
    return dynamic_ids;
}

Functions FilterExecutableFuncs(const Functions& all_funcs,
                                const Variables& roots,
                                const std::set<uint64_t>& dynamic_var_ids,
                                const HasCachedImg& has_cached) {
    std::unordered_set<uint64_t> needed_funcs;
    std::unordered_set<uint64_t> visited_vars;
    Variables stack = roots;
    while (!stack.empty()) {
        Variable v = stack.back();
        stack.pop_back();
        if (!v || !visited_vars.insert(v.id()).second) {
            continue;
        }
        Function creator = v.getCreator();
        if (!creator || IsInlineConst(v)) {
            continue;
        }
        // Static and cached: read the previous iteration's image instead
        if (!dynamic_var_ids.count(v.id()) && has_cached(v)) {
            continue;
        }
        if (needed_funcs.insert(creator.id()).second) {
            for (const auto& x : creator.getInputVars()) {
                stack.push_back(x);
            }
        }
    }

    Functions result;
    for (const auto& f : all_funcs) {
        if (needed_funcs.count(f.id())) {
            result.push_back(f);
        }
    }
    return result;
}

SubStages FilterExecutableSubStages(const SubStages& substages,
                                    const Variables& roots,
                                    const std::set<uint64_t>& dynamic_var_ids,
                                    const HasCachedImg& has_cached) {
    std::unordered_set<uint64_t> needed_vars;
    for (const auto& r : roots) {
        if (r) {
            needed_vars.insert(r.id());
        }
    }

    // Walk in reverse execution order and propagate demand
    std::vector<const SubStage*> kept;
    for (auto it = substages.rbegin(); it != substages.rend(); ++it) {
        const SubStage& ss = *it;
        bool execute = false;
        for (const auto& out : ss.out_vars) {
            if (needed_vars.count(out.id()) &&
                (dynamic_var_ids.count(out.id()) || !has_cached(out))) {
                execute = true;
                break;
            }
        }
        if (!execute) {
            continue;
        }
        kept.push_back(&ss);
        for (const Variables* inputs :
             {&ss.inp_vars, &ss.slt_vars, &ss.tex_vars, &ss.vtx_vars,
              &ss.uif_vars}) {
            for (const auto& v : *inputs) {
                needed_vars.insert(v.id());
            }
        }
    }

    SubStages result;
    for (auto it = kept.rbegin(); it != kept.rend(); ++it) {
        result.push_back(**it);
    }
    return result;
}

}  // namespace dressi
