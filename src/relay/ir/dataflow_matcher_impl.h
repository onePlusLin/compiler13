/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/tvm/relay/dataflow_matcher_impl.h
 * \brief The auxiliary data structure for dataflow matcher.
 */
#ifndef TVM_RELAY_IR_DATAFLOW_MATCHER_IMPL_H_
#define TVM_RELAY_IR_DATAFLOW_MATCHER_IMPL_H_

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/dataflow_pattern.h>
#include <tvm/relay/dataflow_pattern_functor.h>
#include <tvm/relay/expr_functor.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "indexed_graph.h"

namespace tvm {
namespace relay {

class DFPatternMatcher : public DFPatternFunctor<bool(const DFPattern&, const Expr&)> {
 public:
  explicit DFPatternMatcher(const IndexedGraph<Expr>* expr_graph) : expr_graph_(expr_graph) {}
  bool Match(const DFPattern& pattern, const Expr& expr);
  Map<DFPattern, Array<Expr>> GetMemo() { return Map<DFPattern, Array<Expr>>(memo_); }

  const IndexedGraph<Expr>::Node* expr_to_node(const Expr& expr) const {
    return expr_graph_->item_to_node(expr);
  }
  const IndexedGraph<Expr>::Node* index_to_node(size_t index) const {
    return expr_graph_->index_to_node(index);
  }
  size_t size() const { return expr_graph_->size(); }
  const std::unordered_map<DFPattern, Array<Expr>, ObjectPtrHash, ObjectPtrEqual>& memo() const {
    return memo_;
  }
  const IndexedGraph<Expr>& expr_graph() const { return *expr_graph_; }

 protected:
  bool VisitDFPattern(const DFPattern& pattern, const Expr& expr) override;
  bool VisitDFPattern_(const AltPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const AttrPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const CallPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ConstantPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DataTypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DominatorPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const FunctionPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const IfPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const LetPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ShapePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const VarPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) override;

  void ClearMap(size_t watermark);
  bool MatchesPath(const DominatorPatternNode* op, const Expr& expr);
  bool DominatesParent(const DominatorPatternNode* op, const Expr& expr);

  const IndexedGraph<Expr>* expr_graph_;
  std::unordered_map<DFPattern, Array<Expr>, ObjectPtrHash, ObjectPtrEqual> memo_;
  std::vector<DFPattern> matched_nodes_;
  bool memoize_ = true;
};

/*!
 * \brief PatternGrouper does pre-rewriting pattern matching and analysis
 *  PatternGrouper进行预重写模式匹配和分析
 * 
 * This class creates a number of groups of matched expressions, ensures they don't overlap, and
 * returns them to the caller for post-analysis rewriting.
 *这个类创建了许多组匹配的表达式，确保它们不重叠，并且
 *将它们返回给调用者进行分析后重写。
 * 
 * This is primarily needed to support the post-dominator analysis required for dominator pattern
 * matching.
 * 这主要是为了支持dominator模式所需的后dominator分析
 * 匹配
 */
class PatternGrouper {
 public:

  /*! \brief Internal Group class for storing analysis 
  用于存储一个成功匹配的子图的所有分析信息
  */
  struct Group {
    Expr root_node;
    int gid;
    Map<DFPattern, Array<Expr>> matched_nodes;
    std::string name;   //用于存储未来可能生成的划分函数的名字
    Function function;  //用于存储根据此匹配创建的复合函数
    Array<Expr> args;   //存储未来生成的划分函数的参数
  };

  /*! \brief Return the group assignments of expressions 
  返回内部成员 gid_assignments_ 的常量引用。gid_assignments_ 是一个 unordered_map
  */
  inline const std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual>& GetGIDAssignments() {
    return gid_assignments_;
  }
  /*! \brief Group expressions that match the pattern
  这是该类的主要公共接口。它接收一个 DFPattern（要查找的模式）和一个 Expr（要被分析的图），
  然后执行完整的匹配和分组算法，最终返回一个 unordered_map，其中包含了所有找到的组
  */
  const std::unordered_map<int, Group>& GroupMatches(const DFPattern& pattern, const Expr& pre);

 protected:
  /*! \brief 按前序迭代遍历表达式以查找子图
   *
   * If we traverse the graph in post-order, we can run into situtations where a small subgraph will
   * match the pattern. Due to options like AltPattern, a larger subgraph with more nodes later in
   * the graph may also match the pattern. With post-order traversal, we mark the smaller subgraph
   * as matched and fail to catch the larger subgraph. This problem is fixed by using pre-order
   * traversal.
   */
  void VisitExprs();

  /*! \brief Create a group based on a matched expression */
  void CreateGroup(const Expr& expr);

  /*! \brief EmbedConst implements rules for embedding constants into partitioned functions or
   * lifting them into the function arguments.
   *
   * The rules depend on what pattern the ConstantNode matched.
   *
   * The basic rules are:
   *  If the constant matches ExprPattern(relay.const(*)) or a ConstantPattern(), embed the constant
   * in the partitioned function. If the constant matched an AltPattern, recursively check the
   * matched side of the pattern. For any other matching pattern (i.e, wildcard, VarPattern, etc),
   * lift the constant into the arguments of the partitioned function.
   * 这是一个内部决策函数。它判断一个在匹配中遇到的 ConstantNode（常量节点），在未来图划分时，是应该被嵌入 (embed) 到划分出的新函数内部，还是应该被提升 (lift) 为新函数的参数
   */
  bool EmbedConst(const Expr& expr, const DFPattern pattern);
  // Internal State
  DFPattern pattern_; //存储当前要匹配的目标模式
  std::unordered_map<int, Group> groups_; //存储匹配结果，从组 ID (gid) 映射到 Group 结构体
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> gid_assignments_; //存储从 Expr 节点到其所属组 ID 的反向映射。
  DFPatternMatcher* matcher_ = nullptr;//算法所需的内部状态和辅助数据结构
  std::unique_ptr<IndexedGraph<DFPattern>> pattern_graph_;
  int gid_ = 0;       //用于生成唯一组 ID 和图编号的内部计数器。
  int graph_number_ = 0;
};

/*!
 * \brief PatternRewriter rewrites the expression by finding matches and allowing user callback
 * function to rewrite those matches
 *
 * The class uses PatternGrouper to support the dominator pattern.
 * PatternRewriter
 * 通过查找匹配并允许用户回调来重写表达式重写这些匹配项
 *这个类使用PatternGrouper来支持dominator模式。
 */
class PatternRewriter : protected MixedModeMutator {
 public:
  explicit PatternRewriter(IRModule mod) : mod_(mod) {}
  /*! \brief Rewrite can take a number of callbacks and will repeatedly rewrite the graph with the
   * callbacks until it stops changing */
  virtual Expr Rewrite(const Array<DFPatternCallback>& callbacks, const Expr& pre);

 protected:
  virtual Expr DispatchVisitExpr(const Expr& pre);

  IRModule mod_;
  DFPatternCallback callback_;
  std::unordered_map<int, PatternGrouper::Group> groups_;
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> gid_assignments_;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_IR_DATAFLOW_MATCHER_IMPL_H_
