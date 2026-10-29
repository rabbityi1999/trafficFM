"""Independent correction of fractional time-of-day embedding indices.

The source STD-MAE data stores time-of-day as slot/288. Keep that feature in
the encoder input; convert it to an integer ONLY at the embedding lookup.
The completed v1 model and frozen results remain unchanged.
"""
import torch
from . import data_space_fixed


def history_calendar_indices(history, tod_index=1, dow_index=2,
                             slots_per_day=288, tod_format='fraction'):
    if history.ndim != 4 or history.shape[1] <= max(tod_index, dow_index):
        raise ValueError('history must contain speed, time-of-day and day-of-week')
    tod = history[:, tod_index, 0, -1]
    dow = history[:, dow_index, 0, -1]
    if tod_format == 'fraction':
        if not bool((torch.isfinite(tod) & (tod >= 0) & (tod < 1)).all()):
            raise ValueError('fractional time-of-day must be finite and in [0,1)')
        slots = tod * slots_per_day
    elif tod_format == 'index':
        slots = tod
    else:
        raise ValueError('tod_format must be explicitly fraction or index')
    rounded = slots.round()
    if not bool((torch.isfinite(slots) & ((slots-rounded).abs() < 1e-3) &
                 (rounded >= 0) & (rounded < slots_per_day)).all()):
        raise ValueError('time-of-day is not on the configured five-minute grid')
    day = dow.round()
    if not bool((torch.isfinite(dow) & ((dow-day).abs() < 1e-6) & (day >= 0) & (day < 7)).all()):
        raise ValueError('day-of-week must be an integer in [0,7)')
    return rounded.long(), day.long()


class gwnet(data_space_fixed.gwnet):
    training_space = 'standardized_speed_cfm_history_calendar_v2'

    def __init__(self, *args, tod_format='fraction', **kwargs):
        super().__init__(*args, **kwargs)
        if self.tod_emb_dict.num_embeddings != 288:
            raise ValueError('this audited correction requires 288 slots/day')
        self.tod_format = tod_format
        self.register_buffer('_history_calendar_encoding_version', torch.tensor(1))

    def _get_joint_embedding(self, input):
        tod, dow = history_calendar_indices(input, self.in_tod_idx, self.in_dow_idx,
                                            tod_format=self.tod_format)
        time = torch.cat((self.tod_emb_dict(tod), self.dow_emb_dict(dow)), -1)
        batch, _, nodes, _ = input.shape
        return torch.cat((time[:, None].expand(batch, nodes, -1),
                          self.spatial_emb[None].expand(batch, -1, -1)), -1)
