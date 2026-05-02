"""
hook.py — PPA trace generation for NeuroSim v1.5

Fixes from original v1.5:
  1. self.batch_size   → self._cim_args.batch_size
  2. self.name         → self._cim_args.name
  3. dec2val called with correct arguments
  4. Trace format matches the v1.5 C++ backend:
       - Weights: raw float values, shape (in_features, out_features)
         C++ LoadInWeightData normalises and decomposes into cells.
       - Inputs:  unsigned integers, shape (in_features, num_input_vectors)
         C++ LoadInInputData binary-expands them.  One ROW per input element
         because CopyInput indexes rows by feature position.
     The original write_layer pre-encoded both via dec2bin/dec2val and used
     wrong orientation, causing dimension mismatches and segfaults.
"""

import os
import numpy as np
import torch


def write_layer(self, input2d, weight2d):
    """Write PPA trace files for one layer.

    Called from simulate_array (macro.py) after input2d/weight2d have been
    shifted to non-negative and padded with a dummy row/column.  We undo
    those transformations to produce the format the C++ backend expects.
    """

    # ── undo dummy row (input) and dummy column (weight) ───────────
    weight_raw = weight2d[:, :-1]          # remove dummy column
    input_raw  = input2d[:-1, :]           # remove dummy row

    # ── undo the positive shift ────────────────────────────────────
    shift_weight = weight2d[0, -1]         # dummy col fill = shift
    shift_input  = input2d[-1, 0]          # dummy row fill = shift
    weight_raw = weight_raw - shift_weight
    input_raw  = input_raw  - shift_input

    # ── weights → float in [-1, 1] ────────────────────────────────
    weight_max_bound = (2.0 ** (self._cim_args.weight_precision - 1)) - 1.0
    weight_float = weight_raw.float() / weight_max_bound

    # ── inputs → unsigned int, column-oriented ────────────────────
    # C++ CopyInput indexes rows by feature, so each ROW = one input element.
    # input_raw is (batch, in_features); transpose to (in_features, batch).
    input_max = (2.0 ** self._cim_args.input_precision) - 1.0
    input_int = input_raw.clamp(0, input_max).int()

    rows_per_input = input_int.shape[0] // self._cim_args.batch_size \
        if input_int.shape[0] > self._cim_args.batch_size else input_int.shape[0]
    input_int = input_int[0:rows_per_input]        # take one batch
    input_int = input_int.t()                       # (in_features, num_vectors)

    # ── write CSV files ────────────────────────────────────────────
    layer_name  = str(self._cim_args.name)
    model_name  = self._cim_args.model
    record_dir  = './layer_record_' + model_name

    weight_file = record_dir + '/weight' + layer_name + '.csv'
    input_file  = record_dir + '/input'  + layer_name + '.csv'

    np.savetxt(weight_file, weight_float.cpu().numpy(), delimiter=",", fmt='%10.5f')
    np.savetxt(input_file,  input_int.cpu().numpy(),    delimiter=",", fmt='%d')

    with open(record_dir + '/trace_command.sh', 'a') as f:
        f.write(weight_file + ' ' + input_file + ' ')


def make_records(args):
    """Create the layer_record directory and initialise trace_command.sh."""
    record_dir = './layer_record_' + str(args.model)
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    trace_cmd = os.path.join(record_dir, 'trace_command.sh')
    if os.path.exists(trace_cmd):
        os.remove(trace_cmd)

    with open(trace_cmd, 'w') as f:
        f.write('./NeuroSIM/main '
                './NeuroSIM/NetWork_' + str(args.model) + '.csv '
                + str(args.weight_precision) + ' '
                + str(args.input_precision)  + ' '
                + str(args.sub_array[0])     + ' '
                + str(args.parallel_read)    + ' ')

    network_csv = './NeuroSIM/NetWork_' + str(args.model) + '.csv'
    if os.path.exists(network_csv):
        os.remove(network_csv)
