#ifndef DRESSI_PACK_REACTIVE_H
#define DRESSI_PACK_REACTIVE_H

#include <functional>
#include <set>

#include "pack/substage.h"

namespace dressi {

// Variables whose values change this iteration: the seeds (dirty leaves and
// optimizer parameters, which the GPU overwrites every frame) plus everything
// computed from them. `all_funcs` must be creation-id (topologically) sorted.
std::set<uint64_t> ComputeDynamicVarIds(const Functions& all_funcs,
                                        const Variables& seeds);

using HasCachedImg = std::function<bool(const Variable&)>;

// Demand-filters the functions to execute this iteration: walks creators
// from `roots`, stopping at variables that are static (non-dynamic) and
// already cached on the GPU ("RemoveCleanFuncs" of the Dressi paper).
// Preserves topological order.
Functions FilterExecutableFuncs(const Functions& all_funcs,
                                const Variables& roots,
                                const std::set<uint64_t>& dynamic_var_ids,
                                const HasCachedImg& has_cached);

// Substage-granularity counterpart for fast rebuilds
// ("RemoveCleanSubStages"): keeps a substage iff a demanded output is
// dynamic or not yet cached.
SubStages FilterExecutableSubStages(const SubStages& substages,
                                    const Variables& roots,
                                    const std::set<uint64_t>& dynamic_var_ids,
                                    const HasCachedImg& has_cached);

}  // namespace dressi

#endif  // DRESSI_PACK_REACTIVE_H
