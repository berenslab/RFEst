from rfest.EvidenceOpt._base import EmpiricalBayes

__all__ = ['ALD']


class ALD(EmpiricalBayes):
    """

    Automatic Locality Determination (ALD).

    Reference: Park, M., & Pillow, J. W. (2011).

    See also: https://github.com/leaduncker/SimpleEvidenceOpt

    """

    def __init__(self, X, Y, dims, compute_mle=True):
        super().__init__(X, Y, dims, compute_mle, time='ald', space='ald', n_hp_time=4, n_hp_space=4)

    @staticmethod
    def print_progress_header(params):

        print('* Due to space limit, parameters for frequency domain are not printed.')
        if len(params) == 6:
            print('Iter\tσ\tρ\tτ_t\tν_t\tcost')
        elif len(params) == 10:
            print('Iter\tσ\tρ\tτ_t\tν_t\tτ_y\tν_y\tcost')
        elif len(params) == 14:
            print('Iter\tσ\tρ\tτ_t\tν_t\tτ_y\tν_y\tτ_x\tν_x\tcost')

    @staticmethod
    def print_progress(i, params, cost):

        # due to space limit, parameters for \nu are not printed.
        if len(params) == 6:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}'.format(
                i, params[0], params[1], params[2], params[3], cost))
        elif len(params) == 10:
            print('{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}\t{6:1.3f}\t{7:1.3f}'.format(
                i, params[0], params[1], params[2], params[3], params[6], params[7], cost))
        elif len(params) == 14:
            print(
                '{0:4d}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.3f}\t{6:1.3f}\t{7:1.3f}\t{8:1.3f}\t{9:1.3f}'.format(
                    i, params[0], params[1], params[2], params[3], params[6], params[7], params[10], params[11], cost))
