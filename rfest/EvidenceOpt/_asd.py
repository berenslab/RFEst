from rfest.EvidenceOpt._base import EmpiricalBayes

__all__ = ['ASD']


class ASD(EmpiricalBayes):
    """

    Automatic Smoothness Determination (ASD).

    Reference: Sahani, M., & Linden, J. F. (2003). 

    """

    def __init__(self, X, y, dims, compute_mle=True):

        super().__init__(X, y, dims, compute_mle,
                         time='asd', space='asd',
                         n_hp_time=1, n_hp_space=1)

    @staticmethod
    def print_progress_header(params):
        if len(params) == 3:
            print('Iter\tσ\tρ\tδt\tcost')
        elif len(params) == 4:
            print('Iter\tσ\tρ\tδt\tδs\tcost')
        elif len(params) == 5:
            print('Iter\tσ\tρ\tδt\tδy\tδx\tcost')

    @staticmethod
    def print_progress(i, params, cost):

        if len(params) == 3:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}'.format(
                i, params[0], params[1], params[2], cost))
        elif len(params) == 4:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}'.format(
                i, params[0], params[1], params[2], params[3], cost))
        elif len(params) == 5:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}\t{6:1.3f}'.format(
                i, params[0], params[1], params[2], params[3], params[4], cost))
