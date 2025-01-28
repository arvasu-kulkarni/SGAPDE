from pde import *
import warnings
import sys
import time
warnings.filterwarnings('ignore')


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class SGA:  # 最外层
    def __init__(self, num, depth, width, p_var, p_mute, p_rep, p_cro):
        # num: number of PDEs in the pool
        # depth: Maximum depth of term for each PDE
        # width: maximum number ofterms per PDE
        # p_var: probability that a node is u/t/x instead of an operator when generating the tree
        # p_rep: probability that an item of (all) PDEs will be regenerated to replace an original item
        # p_mute: probability of mutation for each node in the tree structure of a PDE
        # p_cro: probability of exchanging term between different PDEs
        self.num = num
        self.p_mute = p_mute
        self.p_cro = p_cro
        self.p_rep = p_rep
        self.eqs = []
        self.mses = []
        self.ratio = 1
        self.repeat_cross = 0
        self.repeat_change = 0
        print('Creating the original pdes in the pool ...')
        for i in range(num*self.ratio): # 循环产生num个pde
            a_pde = PDE(depth, width, p_var)
            a_err, a_w = evaluate_mse(a_pde)
            pde_lib.append(a_pde)
            err_lib.append((a_err, a_w))
            # while a_err < 0.01 or a_err == np.inf:  # MSE太小则直接去除，to avoid u d t
            while a_err < -100 or a_err == np.inf:  # MSE太小则直接去除，to avoid u d t
                print(a_err)
                a_pde = PDE(depth, width, p_var)
                a_err, a_w = evaluate_mse(a_pde)
                pde_lib.append(a_pde)
                err_lib.append((a_err, a_w))
            print('Creating the ith pde, i=', i)
            print('a_pde.visualize():',a_pde.visualize())
            print('evaluate_aic:',a_err)
            self.eqs.append(a_pde)
            self.mses.append(a_err)

        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse) # 从小到大排序，提取出排序的index
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        self.mses, self.eqs = self.mses[0:num], self.eqs[0:num]

        # pdb.set_trace()
    prevbesteq = []
    def run(self, gen=100):
        for i in range(1, gen+1):
            print('here')
            time0 = time.time()
            self.cross_over(self.p_cro)
            time1 = time.time()
            print(f"cross_over took {time1-time0:.2f} seconds")
            self.change(self.p_mute, self.p_rep)
            time2 = time.time()
            print(f"change (mutation) took {time2-time1:.2f} seconds")
            best_eq, best_mse = self.the_best()
            time3 = time.time()
            print(f"the_best took {time3-time2:.2f} seconds")
            print('{} generation best_aic & best Eq: {}, {}'.format(i, best_mse, best_eq.visualize()))
            print('best concise Eq: {}'.format(best_eq.concise_visualize()))
            if i > 25:
                if best_eq.visualize() == self.prevbesteq[-5]:
                    print('The best equation has not changed for 5 generations, stop the iteration')
                    break
            self.prevbesteq.append(best_eq.visualize())
            if best_mse < 0:
                print('We are close to the answer, pay attention')
            print('{} generation repeat cross over {} times and mutation {} times'.
                  format(i, self.repeat_cross, self.repeat_change))
            self.repeat_cross, self.repeat_change = 0, 0

    def the_best(self):
        # print("running the_best")
        argmin = np.argmin(self.mses)
        return self.eqs[argmin], self.mses[argmin]

    def cross_over(self, percentage=0.5): # For example, there are 2n samples in a generation, and the best n samples are first crossed with the best n samples, which produces m new unduplicated samples. Then the best 2n of the 2n+m samples are finally extracted.
        def cross_individual(pde1, pde2):
            time0 = time.time()
            new_pde1, new_pde2 = copy.deepcopy(pde1), copy.deepcopy(pde2)
            w1, w2 = len(pde1.elements), len(pde2.elements)
            ix1, ix2 = np.random.randint(w1), np.random.randint(w2)
            new_pde1.elements[ix1] = pde2.elements[ix2]
            new_pde2.elements[ix2] = pde1.elements[ix1]
            time1 = time.time()
            # print(f"cross_individual took {time1-time0:.2f} seconds")
            return new_pde1, new_pde2

        # Half of the good samples are saved and half of the new samples are cross-generated on this basis
        print('begin crossover')
        time0 = time.time()
        num_ix = int(self.num * percentage)
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)
        
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        copy_mses, copy_eqs = self.mses[0:num_ix], self.eqs[0:num_ix]  # top percentage samples

        new_eqs, new_mse = copy.deepcopy(copy_eqs), copy.deepcopy(copy_mses)
        reo_eqs, reo_mse = copy.deepcopy(copy_eqs), copy.deepcopy(copy_mses)
        random.shuffle(reo_mse)
        random.shuffle(reo_eqs)
        time1 = time.time()
        # print(f"copying took {time1-time0:.2f} seconds")
        for a, b in zip(new_eqs, reo_eqs):
            new_a, new_b = cross_individual(a, b) # Crossover based on good samples
            if new_a.visualize() in pde_lib:
                self.repeat_cross += 1
            else: # The first half of the samples that cross over to produce a new pde are added to the lib and added to all the samples in the current generation
                a_err, a_w = evaluate_mse(new_a)
                pde_lib.append(new_a.visualize())
                err_lib.append((a_err, a_w))
                self.mses.append(a_err)
                self.eqs.append(new_a)

            if new_b.visualize() in pde_lib:
                self.repeat_cross += 1
            else: # 前一半样本交叉产生了新的pde，则加入lib中，并且加入当前代的全部样本中
                b_err, b_w = evaluate_mse(new_b)
                pde_lib.append(new_b.visualize())
                err_lib.append((b_err, b_w))
                self.mses.append(b_err)
                self.eqs.append(new_b)

        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)[0:self.num] # 对当前一代所有的样本和新增非重复样本，整体做一次排序，提取最优的本代样本数个样本。
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]

    def change(self, p_mute=0.05, p_rep=0.3):
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        print(f"self.num is {self.num}")
        for i in range(self.num):
            print(i)
            time0 = time.time()
            # 保留最好的那部分eqs不change，只cross over.
            if i < 1: #保留最好的1个样本，不进行change
                continue
            # print(self.eqs[i].visualize())
            new_eqs[i].mutate(p_mute)
            time1 = time.time()
            print(f"mutate took {time1-time0:.2f} seconds")
            replace_or_not = np.random.choice([False, True], p=([1 - p_rep, p_rep]))
            if replace_or_not:
                new_eqs[i].replace()
            # 检测是否重复
            if new_eqs[i].visualize() in pde_lib:
                self.repeat_change += 1
            else:
                a_err, a_w = evaluate_mse(new_eqs[i])
                pde_lib.append(new_eqs[i].visualize())
                err_lib.append((a_err, a_w))
                self.mses.append(a_err)
                self.eqs.append(new_eqs[i])
                time2 = time.time()
                print(f"evaluate_mse took {time2-time1:.2f} seconds")
        print("here in mutations after all complete")
        new_eqs, new_mse = copy.deepcopy(self.eqs), copy.deepcopy(self.mses)
        sorted_indices = np.argsort(new_mse)[0:self.num] # 对当前一代所有的样本和新增非重复样本，整体做一次排序，提取最优的本代样本数个样本。
        for i, ix in enumerate(sorted_indices):
            self.mses[i], self.eqs[i] = new_mse[ix], new_eqs[ix]


if __name__ == '__main__':
    # np.random.seed(10)
    # a_tree = Tree(max_depth=4, p_var=0.5)
    # print(is_an_equation(a_tree.preorder.split()))

    # pdb.set_trace()

    # pde = PDE(depth=4, max_width=3, p_var=0.5, p_mute=0.1)
    # evaluate_mse(pde)

    # pdb.set_trace()
    sys.stdout = Logger('notes.log', sys.stdout)
    sys.stderr = Logger('notes.log', sys.stderr)
    sga_num = 30
    sga_depth = 4
    sga_width = 5
    sga_p_var = 0.6
    sga_p_mute = 0.75
    sga_p_cro = 0.5
    sga_run = 100

    print('sga_num = ', sga_num)
    print('sga_depth = ', sga_depth)
    print('sga_width = ', sga_width)
    print('sga_p_var = ', sga_p_var)
    print('sga_p_mute = ', sga_p_mute)
    print('sga_p_cro = ', sga_p_cro)
    print('sga_run = ', sga_run)

    sga = SGA(num=sga_num, depth=sga_depth, width=sga_width, p_var=sga_p_var, p_rep=1, p_mute=sga_p_mute, p_cro=sga_p_cro)
    sga.run(sga_run)


