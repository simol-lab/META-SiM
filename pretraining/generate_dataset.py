from multiprocessing import Pool
from multiprocessing import RLock

import numpy as np
from tqdm.auto import tqdm

from smfret.trace_simulator import Simulator
from smfret.trace_simulator import ParameterGenerator
from smfret.trace_simulator import SimulatedTraceSet

from smfret.multi_task_learning import FRETStateTraceSet
from smfret.multi_task_learning import SavedTraceSet
from smfret.multi_task_learning import TwoStateQuickDynamicTraceSet
from smfret.multi_task_learning import TwoStateQuickDynamicNBDTraceSet
from smfret.multi_task_learning import TwoStateQuickDynamicTransitionProbTraceSet
from smfret.multi_task_learning import SelectedSegmentsLowFRETTraceSet
from smfret.multi_task_learning import HighFRETBinaryTraceSet
from smfret.multi_task_learning import LowFRETBinaryTraceSet
from smfret.multi_task_learning import FRETStateHighestTraceSet
from smfret.multi_task_learning import FRETStateLowestTraceSet
from smfret.multi_task_learning import FRETStateCountTraceSet
from smfret.multi_task_learning import NBDTraceSet
from smfret.multi_task_learning import DwellTimeTraceSet
from smfret.multi_task_learning import TransitionRateTraceSet
from smfret.multi_task_learning import NoiseLevelTraceSet
from smfret.multi_task_learning import DonorLifetimeTraceSet
from smfret.multi_task_learning import AcceptorLifetimeTraceSet
from smfret.multi_task_learning import MultistepPhotobleachingTraceSet
from smfret.multi_task_learning import SelectedSegmentsTraceSet
from smfret.multi_task_learning import ReemergentTraceSet
from smfret.multi_task_learning import BackgroundTraceSet
from smfret.multi_task_learning import CoincidentTraceSet
from smfret.multi_task_learning import SingleChannelStateTraceSet
from smfret.multi_task_learning import SingleChannelStateFastTransitionTraceSet
from smfret.multi_task_learning import SingleChannelDwellTimeTraceSet
from smfret.multi_task_learning import SingleChannelBlinkingTraceSet
from smfret.multi_task_learning import SingleChannelLifetimeTraceSet
from smfret.multi_task_learning import SingleChannelNoiseLevelTraceSet
from smfret.multi_task_learning import SingleChannelPhotobleachingStepsFrameLevelTraceSet
from smfret.multi_task_learning import SingleChannelPhotobleachingStepsTraceLevelTraceSet

SIZE=50000
N_POOL=5

def generate_traceset(name):
    """Generates traceset for one class."""

    rng = np.random.default_rng()

    # uses default trace_length = 2000 
    params_gen_base = ParameterGenerator(
        num_states_fn=lambda: rng.integers(low=1, high=3, endpoint=True),
        snr_signal_fn=lambda: rng.uniform(3, 8),
        snr_background_fn=lambda: rng.uniform(3, 8),
        transition_prob_fn=lambda: 1.0 / rng.uniform(2, 200),
        donor_lifetime_fn = lambda: rng.uniform(low=1000, high=2000)
    )

    set_classes = {
        'FRETSTATE': FRETStateTraceSet,
        'FRETSTATECOUNT': FRETStateCountTraceSet,
        'FRETSTATEHIGHEST': FRETStateHighestTraceSet,
        'FRETSTATELOWEST': FRETStateLowestTraceSet,
        'DWELLTIME': DwellTimeTraceSet,
        'NOISELEVEL': NoiseLevelTraceSet,
        'DONORLIFETIME': DonorLifetimeTraceSet,
        'ACCEPTORLIFETIME': AcceptorLifetimeTraceSet,
        'MULTISTEP': MultistepPhotobleachingTraceSet,
        'SELECTEDSEGMENTS': SelectedSegmentsTraceSet,
        'TRANSITIONRATE': TransitionRateTraceSet,
        'REEMERGENT': ReemergentTraceSet,
        'HIGHFRETBINARY': HighFRETBinaryTraceSet,
        'LOWFRETBINARY': LowFRETBinaryTraceSet,
        'NBD': NBDTraceSet, 
        'QUICKTRANSITION': TwoStateQuickDynamicTraceSet,
        'NBDQUICK': TwoStateQuickDynamicNBDTraceSet,
        'TRANSITIONPROBQUICK': TwoStateQuickDynamicTransitionProbTraceSet,
        'LOWFRETSELECTED': SelectedSegmentsLowFRETTraceSet,
        'BACKGROUND': BackgroundTraceSet,
        'COINCIDENCE': CoincidentTraceSet,
        'REEMERGENTBINARY': ReemergentTraceSet,
        'SINGLECHANNELSTATE': SingleChannelStateTraceSet,
        'SINGLECHANNELSTATEFASTTRANSITION': SingleChannelStateFastTransitionTraceSet,
        'SINGLECHANNELDWELLTIME': SingleChannelDwellTimeTraceSet,
        'SINGLECHANNELBLINKING': SingleChannelBlinkingTraceSet,
        'SINGLECHANNELLIFETIME': SingleChannelLifetimeTraceSet,
        'SINGLECHANNELNOISELEVEL': SingleChannelNoiseLevelTraceSet,
        'SINGLECHANNELPHOTOBLEACHINGSTEPSFRAME': SingleChannelPhotobleachingStepsFrameLevelTraceSet,
        'SINGLECHANNELPHOTOBLEACHINGSTEPSTRACE': SingleChannelPhotobleachingStepsTraceLevelTraceSet,
    }

    path = 'saved_dataset/'
    file = path + f'finetune-long-trace-eval/{name}.npz'
    constructor = set_classes[name]
    if name != 'REEMERGENTBINARY':
        constructor(size=SIZE, params_gen=params_gen_base).save(file)
    else:
        constructor(size=SIZE, params_gen=params_gen_base, binary_label=True).save(file)

if __name__ == '__main__':
    set_classes = {
        'FRETSTATE': FRETStateTraceSet,
        'FRETSTATECOUNT': FRETStateCountTraceSet,
        'FRETSTATEHIGHEST': FRETStateHighestTraceSet,
        'FRETSTATELOWEST': FRETStateLowestTraceSet,
        'DWELLTIME': DwellTimeTraceSet,
        'NOISELEVEL': NoiseLevelTraceSet,
        'DONORLIFETIME': DonorLifetimeTraceSet,
        'ACCEPTORLIFETIME': AcceptorLifetimeTraceSet,
        'MULTISTEP': MultistepPhotobleachingTraceSet,
        'SELECTEDSEGMENTS': SelectedSegmentsTraceSet,
        'TRANSITIONRATE': TransitionRateTraceSet,
        'REEMERGENT': ReemergentTraceSet,
        'HIGHFRETBINARY': HighFRETBinaryTraceSet,
        'LOWFRETBINARY': LowFRETBinaryTraceSet,
        'NBD': NBDTraceSet, 
        'QUICKTRANSITION': TwoStateQuickDynamicTraceSet,
        'NBDQUICK': TwoStateQuickDynamicNBDTraceSet,
        'TRANSITIONPROBQUICK': TwoStateQuickDynamicTransitionProbTraceSet,
        'LOWFRETSELECTED': SelectedSegmentsLowFRETTraceSet,
        'BACKGROUND': BackgroundTraceSet,
        'COINCIDENCE': CoincidentTraceSet,
        'REEMERGENTBINARY': ReemergentTraceSet,
        'SINGLECHANNELSTATE': SingleChannelStateTraceSet,
        'SINGLECHANNELSTATEFASTTRANSITION': SingleChannelStateFastTransitionTraceSet,
        'SINGLECHANNELDWELLTIME': SingleChannelDwellTimeTraceSet,
        'SINGLECHANNELBLINKING': SingleChannelBlinkingTraceSet,
        'SINGLECHANNELLIFETIME': SingleChannelLifetimeTraceSet,
        'SINGLECHANNELNOISELEVEL': SingleChannelNoiseLevelTraceSet,
        'SINGLECHANNELPHOTOBLEACHINGSTEPSFRAME': SingleChannelPhotobleachingStepsFrameLevelTraceSet,
        'SINGLECHANNELPHOTOBLEACHINGSTEPSTRACE': SingleChannelPhotobleachingStepsTraceLevelTraceSet,
    }
    
    with Pool(N_POOL, initargs=(RLock(),), initializer=tqdm.set_lock) as p:
        p.map(generate_traceset, set_classes.keys())