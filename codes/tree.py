from setup import *
import copy
import numpy as np
import datetime

class Node:
    """
        1. depth: depth of the node
        2. idx: the first node of the current depth
        3. parent_idx: parent_idx is the first node in the previous level.
        4. name: node details
        5. child_num: how many children the node has (unary or double)
        6. child_st: the node's current child from the first node of the next level up
        7. var: node variable/operation
        8. cache: Retain data from operations to date.
        9. status: Initialized to child_num, used to record traversal status.
        10. full: full information, expressed as OP or VAR.
    """
    def __init__(self, depth, idx, parent_idx, name, full, child_num, child_st, var):
        self.depth = depth
        self.idx = idx
        self.parent_idx = parent_idx

        self.name = name
        self.child_num = child_num
        self.child_st = child_st
        self.status = self.child_num
        self.var = var
        self.full = full
        self.cache = copy.deepcopy(var)

    def __str__(self): # 提供节点详情
        return ' '.join(str(v) for v in (self.name, self.depth, self.parent_idx, self.idx, type(self.cache), type(self.var)))

    def reset_status(self): # 初始化status
        self.status = self.child_num


class Tree: #对应于pde中的一个term
    def __init__(self, max_depth, p_var):
        self.max_depth = max_depth
        self.tree = [[] for i in range(max_depth)]
        self.preorder, self.inorder = None, None

        root = ROOT[np.random.randint(0, len(ROOT))] # 随机产生初始root（一种OPS）# e.g. ['sin', 1, np.sin], ['*', 2, np.multiply] 
        node = Node(depth=0, idx=0, parent_idx=None, name=root[0], var=root[2], full=root,
                    child_num=int(root[1]), child_st=0) # 设置初始节点Node
        self.tree[0].append(node) # 初始节点
        # print(f"new tree, root is {node.name} id 0")
        depth = 1
        # print(f"max depth is {max_depth}")
        while depth < max_depth: # next_cnt = 0
            next_cnt = 0 #child_st = next_cnt, child_st: the next level is the current child node from the first node onwards
            # Correspond to each parent node by continuing to generate their children
            for parent_idx in range(len(self.tree[depth - 1])): # A node at a certain depth in a tree can have more than one child, so it is possible to have more than one node at a given depth
                parent = self.tree[depth - 1][parent_idx] # extract the corresponding operator (a node) at the corresponding depth
                if parent.child_num == 0: # If the current node has no children, skip the rest of the loop and continue to the next round of loops
                    continue
                for j in range(parent.child_num): # If the current node has no children, skip the rest of the loop and continue.
                    # rule 1: parent var is d and j is 1, must ensure right child is x
                    if parent.name in {'d', 'd^2'} and j == 1: # j == 0 for d's left node, j == 1 for d's right node
                        node = den[np.random.randint(0, len(den))] # Randomly generate a denominator for differential operations, typically xyt
                        node = Node(depth=depth, idx=len(self.tree[depth]), parent_idx=parent_idx, name=node[0],
                                    var=node[2], full=node, child_num=int(node[1]), child_st=None)
                        self.tree[depth].append(node)
                    # rule 2: bottom level must be var, not op
                    elif depth >= max_depth - 1:
                        # print("here VAR")
                        node = VARS[np.random.randint(0, len(VARS))]
                        node = Node(depth=depth, idx=len(self.tree[depth]), parent_idx=parent_idx, name=node[0],
                                    var=node[2], full=node, child_num=int(node[1]), child_st=None)
                        self.tree[depth].append(node)
                    else:
                    # rule 3: not the bottom layer, p_var probability produces var, if var is produced, child_st is None. when oops is produced, the corresponding child_st when the corresponding node is produced is obtained by computation. so that it corresponds to the child node corresponding to that oops in the next level.
                        if np.random.random() <= p_var:
                            node = VARS[np.random.randint(0, len(VARS))]
                            node = Node(depth=depth, idx=len(self.tree[depth]), parent_idx=parent_idx, name=node[0],
                                        var=node[2], full=node, child_num=int(node[1]), child_st=None)
                            self.tree[depth].append(node)
                        else:
                            node = OPS[np.random.randint(0, len(OPS))]
                            node = Node(depth=depth, idx=len(self.tree[depth]), parent_idx=parent_idx, name=node[0],
                                        var=node[2], full=node, child_num=int(node[1]), child_st=next_cnt)
                            next_cnt += node.child_num
                            self.tree[depth].append(node)
                        
                    # print(f"node {node.name} id {len(self.tree[depth])} added at depth {depth} with parent {parent.name} id {parent_idx}")
            depth += 1

        ret = []
        dfs(ret, self.tree, depth=0, idx=0)
        self.preorder = ' '.join([x for x in ret])
        model_tree = copy.deepcopy(self.tree)
        self.inorder = tree2str_merge(model_tree)
        # print(model_tree)

        # print(self.preorder)
        # print(self.inorder)
        # print('---------------')

    def mutate(self, p_mute): #Replace a node in the original tree directly with a node of the same type, so that subsequent positions don't need to be regenerated (analogous to replacing a gene, rather than regenerating the subsequent gene sequence, which has physical implications and is easy to implement)
        # print(f'mutating!! {str(datetime.datetime.now())}')
        global see_tree
        see_tree = copy.deepcopy(self.tree)
        depth = 1
        while depth < self.max_depth:
            next_cnt = 0
            idx_this_depth = 0  # How many nodes in this depth?
            for parent_idx in range(len(self.tree[depth - 1])):
                parent = self.tree[depth - 1][parent_idx]
                if parent.child_num == 0:
                    continue
                for j in range(parent.child_num):  # The jth child node of parent
                    not_mute = np.random.choice([True, False], p=([1 - p_mute, p_mute]))
                    # rule 1: Skip if no mutation
                    if not_mute:
                        next_cnt += self.tree[depth][parent.child_st + j].child_num
                        continue
                    # Type of current node
                    current = self.tree[depth][parent.child_st + j]
                    temp = self.tree[depth][parent.child_st + j].name
                    num_child = self.tree[depth][parent.child_st + j].child_num  # number of children of the current mutation node
                    # print('mutate!')
                    # will have to separate case where node is du/d[x] and where node is u or 0, which are not in den
                    if num_child == 0: # leaf node
                        # node = VARS[np.random.randint(0, len(VARS))] # rule 2: Leaf nodes must be var, not op
                        # while node[0] == temp or (parent.name in {'d', 'd^2'} and node[0] not in den[:, 0]):# rule 3: If the results are duplicated before and after compilation, or if the nodes of d are not in den (i.e., there are objects that cannot be derived), then reextract the
                        #     if simple_mode and parent.name in {'d', 'd^2'} and node[0] == 'x': # simple_mode in which the derivative with respect to x is encountered, stopping the variation directly
                        #         break                            
                        #     node = VARS[np.random.randint(0, len(VARS))] # Redraw a vars
                        if parent.name in ('d', 'd^2'):
                            # must be from den, so either x, y
                            node = den[np.random.randint(0, len(den))]
                            iter = 0
                            while node[0] == temp:
                                iter += 1
                                assert iter < 100, "too many iterations for parent name d, d^2"
                                node = den[np.random.randint(0, len(den))]
                        else:
                            # just operand variable, so u or 0
                            if len(VARS) > 1:
                                node = VARS[np.random.randint(0, len(VARS))]
                                iter = 0
                                while node[0] == temp:
                                    iter += 1
                                    assert iter < 100, "too many iterations for VARS"
                                    node = VARS[np.random.randint(0, len(VARS))]
                            else:
                                # no change, but this causes problems p_mutation is now less than what it was set to
                                print("WARNING - len(VARS) = 1!! P(mutation) is lower than expected!")
                                node = VARS[0]
                        new_node = Node(depth=depth, idx=idx_this_depth, parent_idx=parent_idx, name=node[0],
                                     var=node[2], full=node, child_num=int(node[1]), child_st=None)
                        self.tree[depth][parent.child_st + j] = new_node #替换成变异的节点
                    else: # non-leaf node
                        if num_child == 1:
                            node = OP1[np.random.randint(0, len(OP1))]
                            iter = 0
                            while node[0] == temp:  # Avoiding duplication
                                iter += 1
                                node = OP1[np.random.randint(0, len(OP1))]
                                assert iter < 100, "OP1 iterations"
                        elif num_child == 2:
                            node = OP2[np.random.randint(0, len(OP2))]
                            right = self.tree[depth + 1][current.child_st + 1].name
                            iter = 0
                            while node[0] == temp or (node[0] in {'d', 'd^2'} and right not in den[:, 0]):# rule 4: Avoid duplication, avoid generating d to disrupt tree structure (right child node of new d is not x)
                                iter += 1
                                node = OP2[np.random.randint(0, len(OP2))]
                                assert iter < 100, "OP2 iterations"
                        else:
                            raise NotImplementedError("Error occurs!")

                        new_node = Node(depth=depth, idx=idx_this_depth, parent_idx=parent_idx, name=node[0],
                                    var=node[2], full=node, child_num=int(node[1]), child_st=next_cnt)
                        next_cnt += new_node.child_num
                        self.tree[depth][parent.child_st + j] = new_node
                    idx_this_depth += 1
            depth += 1

        ret = []
        dfs(ret, self.tree, depth=0, idx=0)
        self.preorder = ' '.join([x for x in ret])
        model_tree = copy.deepcopy(self.tree)
        self.inorder = tree2str_merge(model_tree)

        # print(self.preorder)
        # print(self.inorder)


def dfs(ret, a_tree, depth, idx): #辅助前序遍历，产生一个描述这个tree的名称序列（ret）
    # print(depth, idx)  # 深度优先遍历的顺序
    # print(len(a_tree), depth)
    node = a_tree[depth][idx]
    ret.append(node.name) # 记录当前操作
    for ix in range(node.child_num):
        if node.child_st is None:
            continue
        else:
            # print(a_tree[depth][idx].name)
            dfs(ret, a_tree, depth+1, node.child_st + ix) #进入下一层中下一个节点对应的子节点


def tree2str_merge(a_tree):
    # print("new call to func")
    for i in range(len(a_tree) - 1, 0, -1):
        # print()
        for node in a_tree[i]:
            # print(node)
            if node.status == 0:
                if a_tree[node.depth-1][node.parent_idx].status == 1:
                    if a_tree[node.depth-1][node.parent_idx].child_num == 2:
                        a_tree[node.depth-1][node.parent_idx].name = a_tree[node.depth-1][node.parent_idx].name + ' ' + node.name + ')'
                    else:
                        a_tree[node.depth-1][node.parent_idx].name = '( ' + a_tree[node.depth-1][node.parent_idx].name + ' ' + node.name + ')'
                elif a_tree[node.depth-1][node.parent_idx].status > 1:
                    a_tree[node.depth-1][node.parent_idx].name = '(' + node.name + ' ' + a_tree[node.depth-1][node.parent_idx].name
                a_tree[node.depth-1][node.parent_idx].status -= 1
    return a_tree[0][0].name


# class Point:
#     def __init__(self, idx, name, child_num, child_idx=[]):
#         """
#             1. idx: 当前序列的第几个节点
#             2. parent_idx: 父节点是第几个节点
#             3. name: 节点名称
#             4. child_num: 节点拥有几个孩子节点
#             5. child_idx: 孩子节点是序列的第几个
#         """
#         self.idx = idx
#         self.name = name
#         self.child_num = child_num
#         self.child_idx = child_idx

#     def __str__(self):
#         return self.name

#     def add_child(self, ix):
#         self.child_idx.append(ix)


# def is_an_equation(seq):  # e.g. (+ u - u u)
#     def split(seq, idx):
#         # last element is an op
#         if idx >= len(seq): return np.inf

#         # idx is the current node
#         op = ALL[:, 0]
#         root = ALL[np.where(op == seq[idx])][0]
#         node = Point(idx=idx, name=root[0], child_num=int(root[1]))

#         if node.child_num != 0:
#             node.child_idx.append(idx + 1)  # might be wrong for the last node, not fatal though
#             new_idx = split(seq, idx + 1)  # first child
#             if node.child_num != 1:  # other children
#                 node.child_idx.append(new_idx)
#                 new_idx = split(seq, new_idx)
#             return new_idx

#         return idx + 1

#     idx = 0
#     end_idx = split(seq, idx)
#     if end_idx != len(seq):
#         return False
#     return True


if __name__ == '__main__':
    tree = Tree(max_depth=4, p_var=0.5)
    print(tree.inorder)
    tree.mutate(p_mute=1)
    print(tree.inorder)

