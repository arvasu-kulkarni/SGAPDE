import time
time0 = time.time()
from tree import *
from PDE_find import Train
from configure import aic_ratio
from setup import default_names, num_default, u
import warnings
import numpy as np
time1 = time.time()
print(f"Importing in pde.py took {time1-time0:.2f} seconds")
warnings.filterwarnings('ignore')


class PDE:
    def __init__(self, depth, max_width, p_var):
        self.depth = depth
        self.p_var = p_var
        self.W = np.random.randint(max_width)+1  # 1 -- width
        self.elements = []
        for i in range(0, self.W):
            # 产生W个tree，也就是W个项
            one_tree = Tree(depth, p_var)
            # while 'd u t' in tree.preorder:# 没用，挡不住如(sin x + u) d t；不如直接看mse，太小就扔掉
            #     tree = Tree(depth, p_var)
            self.elements.append(one_tree)

    def mutate(self, p_mute):
        for i in range(0, self.W):  # 0 -- W-1
            self.elements[i].mutate(p_mute)

    def replace(self): # 直接产生一个新的tree，替换pde中的一项
        # print('replace!')
        one_tree = Tree(self.depth, self.p_var)
        ix = np.random.randint(self.W)  # 0 -- W-1
        if len(self.elements) == 0:
            NotImplementedError('replace error')
        self.elements[ix] = one_tree

    def visualize(self): # 写出SGA产生的项的形式，包含产生的所有项，未去除系数小的项。
        name = ''
        for i in range(len(self.elements)):
            if i != 0:
                name += '+'
            name += self.elements[i].inorder
        return name

    def concise_visualize(self): # 写出所有项的形式，包含固定候选集和SGA，且包含系数。会区分是来自于固定候选集的还是来自于SGA生成的候选集的。如果是来自于SGA生成的候选集，需要用inorder来写出可理解的项。
        name = ''
        elements = copy.deepcopy(self.elements)
        elements, coefficients = evaluate_mse(elements, True)
        coefficients = coefficients[:, 0]
        # print(len(elements), len(coefficients))
        for i in range(len(coefficients)):
            if np.abs(coefficients[i]) < 1e-4: # 忽略过于小的系数
                continue
            if i != 0 and name != '':
                name += ' + '
            name += str(round(np.real(coefficients[i]), 4))
            if i < num_default: # num_default中为一定包含的候选集
                name += default_names[i]
            else:
                name += elements[i-num_default].inorder # element是SGA生成的候选集
        return name

def printTree(tree_list):
    for i in range(len(tree_list)):
        name = ' '.join(node.name for node in tree_list[i])
        print(name)

def evaluate_mse(a_pde, is_term=False):
    if is_term:
        terms = a_pde
    else:
        terms = a_pde.elements
    terms_values = np.zeros((u.size, len(terms)))
    # print(terms_values.shape)
    delete_ix = []
    time0 = time.time()
    for ix, term in enumerate(terms):
        # print("new tree")
        tree_list = term.tree
        max_depth = len(tree_list)

        # Search the penultimate layer first, and work your way up to the top, excluding the empty layer at the bottom.
        for i in range(2, max_depth+1):
            # If the bottom layer is empty, then it must not be a non-empty penultimate layer.
            if len(tree_list[-i+1]) == 0:
                continue
            else: # This is the non-empty penultimate level, look at it one node at a time.
                for j in range(len(tree_list[-i])): # If this node has no children, continue.
                    # If this node has no children, continue looking at the nodes to the right.
                    if tree_list[-i][j].child_num == 0: # If this node has no children, continue to the right node.
                        continue

                    # If this node has a child, use your own operator on the child's cache
                    elif tree_list[-i][j].child_num == 1:
                        child_node = tree_list[-i+1][tree_list[-i][j].child_st]
                        tree_list[-i][j].cache = tree_list[-i][j].cache(child_node.cache)
                        child_node.cache = child_node.var # reset

                    # This node has one or two children, use your own operators on the two children's cache
                    elif tree_list[-i][j].child_num == 2:
                        child1 = tree_list[-i+1][tree_list[-i][j].child_st]
                        child2 = tree_list[-i+1][tree_list[-i][j].child_st+1]

                        if tree_list[-i][j].name in {'d', 'd^2'}:
                            what_is_denominator = child2.name
                            if what_is_denominator == 't':
                                tmp = dt
                            elif what_is_denominator == 'x':
                                tmp = dx
                            elif what_is_denominator == 'y':
                                tmp = dy
                            else:
                                raise NotImplementedError()

                            if not isfunction(tree_list[-i][j].cache):
                                pdb.set_trace()
                                tree_list[-i][j].cache = tree_list[-i][j].var

                            try:
                                tree_list[-i][j].cache = tree_list[-i][j].cache(child1.cache, tmp, what_is_denominator)
                            except:
                                print(f"Error in shape, child cache shape is {child1.cache.shape}")
                                print(f"tree is \n{tree2str_merge(tree_list)}")
                                raise Exception

                        else:
                            # print('Before error - ')
                            # print(tree_list[-i][j], tree_list[-i][j].cache)
                            if isfunction(child1.cache) or isfunction(child2.cache):
                                pdb.set_trace()
                            tree_list[-i][j].cache = tree_list[-i][j].cache(child1.cache, child2.cache)
                        child1.cache, child2.cache = child1.var, child2.var # reset

                else:
                        NotImplementedError()
        # print(f"cache shape at root - {tree_list[0][0].cache.shape}")
        if not any(tree_list[0][0].cache.reshape(-1)): # if all zeros, not convergent and meaningless
            delete_ix.append(ix)
            tree_list[0][0].cache = tree_list[0][0].var # reset the buffer pool
            # print('0')
            # pdb.set_trace()
        else:
            # print(f'cache shape - {tree_list[0][0].cache.shape}')
            terms_values[:, ix:ix+1] = tree_list[0][0].cache.flatten().reshape(-1, 1) # Record that term grouped together
            tree_list[0][0].cache = tree_list[0][0].var # reset the buffer pool
            # print('not 0')
            # pdb.set_trace()
        
    time1 = time.time()
    print(f"evaluate_mse took {time1-time0:.2f} seconds")
    move = 0
    for ixx in delete_ix:
        if is_term:
            terms.pop(ixx - move)
        else:
            a_pde.elements.pop(ixx-move)
            a_pde.W -= 1  # 实际宽度减一
        terms_values = np.delete(terms_values, ixx-move, axis=1)
        move += 1  # pop以后index左移
    time2 = time.time()
    print(f"deleting took {time2-time1:.2f} seconds")

    # 检查是否存在inf或者nan，或者terms_values是否被削没了
    if False in np.isfinite(terms_values) or terms_values.shape[1] == 0:
        # pdb.set_trace()
        error = np.inf
        aic = np.inf
        w = 0

    else:
        # 2D --> 1D
        # terms_values = np.hstack((default_terms, terms_values))
        # print(f'ut shape - {ut.shape}')
        w, loss, mse, aic = Train(terms_values, ut.reshape(-1, 1), 0, 1, aic_ratio)

    time3 = time.time()
    print(f"Train took {time3-time2:.2f} seconds")

    if is_term:
        return terms, w
    else:
        return aic, w


if __name__ == '__main__':
    pde = PDE(depth=4, max_width=3, p_var=0.5)
    evaluate_mse(pde)
    pde.mutate(p_mute=0.1)
    pde.replace()

