#ifndef DRESSI_VARIABLE_H
#define DRESSI_VARIABLE_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "dressi/types.h"

namespace dressi {

struct VariableNode;
class Function;

// Variable object: a node of the computational graph representing a 2D image.
// Thin value-semantics handle over a shared node (handle/body idiom).
class Variable {
public:
    Variable(VType vtype = FLOAT, ImgSize img_size = {1, 1});
    Variable(std::nullptr_t) {}  // null handle for "no gradient"

    explicit operator bool() const { return m_node != nullptr; }
    bool operator==(const Variable& o) const { return m_node == o.m_node; }
    bool operator!=(const Variable& o) const { return m_node != o.m_node; }
    uint64_t id() const;

    // Getters / setters
    VType getVType() const;
    void setVType(VType vtype);
    ImgSize getImgSize() const;
    void setImgSize(ImgSize img_size);
    Function getCreator() const;
    void setCreator(const Function& creator);
    std::vector<Function> getUsers() const;
    void addUser(const Function& user);
    bool getRequiresGrad() const;
    void setRequiresGrad(bool req_grad);
    void setRequiresGradRecursively(bool req_grad = true);
    bool IsDirty() const;
    void setDirty(bool is_dirty);
    void setDirtyRecursively(bool is_dirty);

private:
    friend struct NodeAccess;
    explicit Variable(std::shared_ptr<VariableNode> node)
        : m_node(std::move(node)) {}
    std::shared_ptr<VariableNode> m_node;
};

using Variables = std::vector<Variable>;

}  // namespace dressi

namespace dressi {

// Orders by creation id; operator< on Variables is reserved for building
// comparison graphs (e.g. `0.f < edge_dist`). Not a std::less specialization:
// libc++ rewrites std::less<Key> in map/set to std::less<void>, which would
// bypass it and call that operator<.
struct VariableLess {
    bool operator()(const Variable& a, const Variable& b) const {
        const uint64_t l = a ? a.id() : 0;
        const uint64_t r = b ? b.id() : 0;
        return l < r;
    }
};

template <typename T>
using VarMap = std::map<Variable, T, VariableLess>;
using VarSet = std::set<Variable, VariableLess>;

}  // namespace dressi

template <>
struct std::hash<dressi::Variable> {
    size_t operator()(const dressi::Variable& v) const {
        return v ? std::hash<uint64_t>()(v.id()) : 0;
    }
};

#endif  // DRESSI_VARIABLE_H
