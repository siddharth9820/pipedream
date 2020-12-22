# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import print_function

import argparse
from collections import OrderedDict, defaultdict
import csv
import math
import os

import sys
sys.path.append("..")
import graph
import utils
import numpy as np
import json

USE_MY = True


def compute_partitioning(compute_times, activation_sizes, parameter_sizes,
                         input_activation_sizes, output_activation_sizes, all_predecessor_ids,
                         num_machines, num_machines_within_machine,
                         bandwidth, pipeline_limit, unsplittables, final_level=True):
    A = []
    for i in range(len(compute_times)):
        row_A = []
        for j in range(len(compute_times[0])):
            row_row_A = []
            for m in range(num_machines):
                row_row_row_A = []
                for p in range(pipeline_limit + 1):
                    row_row_row_A.append((None, None, None)) 
                row_row_A.append(row_row_row_A)
            row_A.append(row_row_A)
        A.append(row_A)

    
    for i in range(len(compute_times)):
        for j in range(i, len(compute_times[0])):
            cum_activation_size = activation_sizes[i][j]
            cum_parameter_size = parameter_sizes[i][j]
            input_activation_size = input_activation_sizes[i]
            for pl in range(1, pipeline_limit+1):
                max_m = 1 if straight_pipeline else num_machines
                for m in range(max_m):
                    sub_level_pl = pl // (m+1) 
                    if pl % (m+1) != 0:
                        sub_level_pl += 1

                    cum_compute_time = compute_times[i][j][sub_level_pl]                
                    data_parallel_communication_time = (4 * (m) * cum_parameter_size) / (bandwidth * (m + 1))
                    data_parallel_communication_time /= num_machines_within_machine

                    if (cum_compute_time is None) or (i in unsplittables) or (j+1 in unsplittables):
                        A[i][j][m][pl] = (None, None, None)
                    else:
                        A[i][j][m][pl] = (sum([cum_compute_time,
                                    data_parallel_communication_time]) / (m+1), None, (m+1))

                    if (m+1)*num_machines_within_machine >= pl :
                        break
    
    # print("sanity check for i=0, j=3...")
    # for pl in range(1, pipeline_limit+1):
    #     max_m = 1 if straight_pipeline else min(pl,num_machines)
    #     for m in range(max_m):
    #         print("With {} machines pl {} -  {}".format(m+1, pl, A[0][3][m][pl][0]))
    

    min_machines = 1 
    max_i = len(compute_times) if not final_level else 1
    for i in range(max_i):
        if i in unsplittables:
            continue
        for m in range(min_machines, num_machines):
            for j in range(i+1, len(compute_times[0])):
                if j+1 in unsplittables:
                    continue
                for pl in range(1, pipeline_limit+1):
                    (min_pipeline_time, optimal_split, optimal_num_machines) = A[i][j][m][pl]
                    if use_fewer_machines and m > 0 and (
                        min_pipeline_time is None or (A[i][j][m-1][pl][0] and (A[i][j][m-1][pl][0] < min_pipeline_time) ) ):
                        (min_pipeline_time, optimal_split, optimal_num_machines) = A[i][j][m-1][pl]
                    for k in all_predecessor_ids[j]:
                        if i > 0 and k in all_predecessor_ids[i-1]:
                            continue
                        if k+1 in unsplittables:
                            continue
                        max_m_prime = 2 if straight_pipeline else (m+1)
                        for m_prime in range(1, max_m_prime):
                            input_transfer_time = (2.0 * output_activation_sizes[k]) / \
                                (bandwidth * m_prime)
                            output_transfer_time = None
                            if j < len(output_activation_sizes) -1:
                                output_transfer_time = (2.0 *
                                    output_activation_sizes[j]) / (bandwidth * m_prime)

                            sub_level_pl = pl // (m_prime) 
                            if pl % (m_prime) != 0:
                                sub_level_pl += 1
                            last_stage_time = compute_times[k+1][j][sub_level_pl]

                            if last_stage_time is None:
                                continue
                            
                            # A DP partition of [k+1,j] with repl factor m_prime is possible with pipeline limit pl 

                            last_stage_parameter_size = parameter_sizes[k+1][j]

                            last_stage_time = sum([last_stage_time,
                                                ((4 * (m_prime - 1) *
                                                    last_stage_parameter_size) / (bandwidth * m_prime))])
                            last_stage_time /= m_prime

                            
                            if A[i][k][m-m_prime][pl][0] is None:
                                continue
                            
                            # A solution for [i,k] with m-m_prime + 1 machines exists with pipeline limit pl 

                            pipeline_time = max(A[i][k][m-m_prime][pl][0], last_stage_time)
                            
                            if activation_compression_ratio is not None:
                                input_transfer_time /= activation_compression_ratio
                                if output_transfer_time is not None:
                                    output_transfer_time /= activation_compression_ratio
                                pipeline_time = max(pipeline_time, input_transfer_time)
                                if output_transfer_time is not None:
                                    pipeline_time = max(pipeline_time, output_transfer_time)
                            if min_pipeline_time is None or min_pipeline_time > pipeline_time:
                                optimal_split = (k, m-m_prime)
                                optimal_num_machines = m_prime
                                min_pipeline_time = pipeline_time
                            
                            if m_prime * num_machines_within_machine >= pl:
                                break

                    A[i][j][m][pl] = (min_pipeline_time, optimal_split, optimal_num_machines)

    return A

def analyze_partitioning(A, states, start, end, network_bandwidth, num_machines, pipeline_limit, input_activation_sizes,
                         activation_compression_ratio, print_configuration, verbose):
    metadata = A[start][end-1][num_machines-1][pipeline_limit]
    next_split = metadata[1]
    remaining_machines_left = num_machines
    splits = []
    replication_factors = []
    activation_sizes = []
    prev_split = end - 1
    #fixed the bug for wrong prev_split initially 
    while next_split is not None:
        num_machines_used = metadata[2]
        if verbose:
            print("-------------------------------------")
            print("Number of machines used: %d..." % num_machines_used)
            print("Split between layers %d and %d..." % (next_split[0], next_split[0] + 1))
            print("Split before antichain %s..." % (states[next_split[0]+1].antichain))
        splits.append(next_split[0]+1)
        compute_time = states[prev_split].compute_time - \
            states[next_split[0]].compute_time
        parameter_size = states[prev_split].parameter_size - \
            states[next_split[0]].parameter_size
        activation_size = states[prev_split].activation_size - \
            states[next_split[0]].activation_size
        activation_sizes.append(activation_size)
        dp_communication_time = (4 * (num_machines_used - 1) * parameter_size) \
            / (network_bandwidth * num_machines_used)
        pp_communication_time_input = (
            2.0 * states[next_split[0]].output_activation_size *
            (1.0 / float(num_machines_used))) / network_bandwidth
        pp_communication_time_output = (
            2.0 * states[prev_split].output_activation_size *
            (1.0 / float(num_machines_used))) / network_bandwidth
        if activation_compression_ratio is not None:
            pp_communication_time_input /= activation_compression_ratio
            pp_communication_time_output /= activation_compression_ratio
        if activation_compression_ratio is None:
            pp_communication_time_input = 0.0
            pp_communication_time_output = 0.0

        compute_time /= num_machines_used
        dp_communication_time /= num_machines_used

        if verbose:
            print(("Compute time = %f, Data-parallel communication time = %f, "
                   "Pipeline-parallel communication time = %f...") % (
                compute_time, dp_communication_time,
                max(pp_communication_time_input, pp_communication_time_output)))
        prev_split = splits[-1] - 1 
        metadata = A[start][next_split[0]][next_split[1]][pipeline_limit]
        next_split = metadata[1]
        replication_factors.append(num_machines_used)
        remaining_machines_left -= num_machines_used
    if verbose:
        print("-------------------------------------")
        print("Number of machines used: %d..." % metadata[2])

    num_machines_used = metadata[2]
    remaining_machines_left -= num_machines_used
    compute_time = states[prev_split].compute_time
    parameter_size = states[prev_split].parameter_size
    dp_communication_time = ((4 * (num_machines_used - 1) * parameter_size) /
                             (network_bandwidth * num_machines_used))
    compute_time /= num_machines_used
    dp_communication_time /= num_machines_used
    activation_sizes.append(states[prev_split].activation_size)
   
    if verbose:
        print("Compute time = %f, Data-parallel communication time = %f..." %
              (compute_time, dp_communication_time))
        print("-------------------------------------")
    if print_configuration:
        print("Number of machines in budget not used: %d..." %
              remaining_machines_left)
        print()
        print("(Split start, split end) / compute time taken per stage "
              "/ replication factor per stage / Memory  Occupancy:")
    prev_split = start
    splits.reverse()
    splits.append(end)
    replication_factors.append(num_machines_used)
    replication_factors.reverse()
    activation_sizes.reverse()

    for i in range(len(splits)):
        time = 0.0
        act = 0
        if prev_split > 0:
            time = states[splits[i]-1].compute_time - states[prev_split-1].compute_time
            act = states[splits[i]-1].activation_size - states[prev_split-1].activation_size
            par = states[splits[i]-1].parameter_size - states[prev_split-1].parameter_size
        else:
            time = states[splits[i]-1].compute_time
            act = states[splits[i]-1].activation_size
            par = states[splits[i]-1].parameter_size
        ip = input_activation_sizes[prev_split]
        if print_configuration:
            mul_fac = pipeline_limit // replication_factors[i]
            if pipeline_limit % replication_factors[i] !=0:
                mul_fac += 1
            print((prev_split, splits[i]), "{0:.4f}".format(time), replication_factors[i], mul_fac*(act+par)/1e6," MB")
        prev_split = splits[i]
    if print_configuration:
        print()
    return splits[:-1], replication_factors

def main(all_num_machines, profile_filename, network_bandwidths, memory_size,
         straight_pipeline, use_memory_constraint, use_fewer_machines,
         activation_compression_ratio, output_directory, pipeline_limit,
         print_configuration=True, verbose=False):
    gr = graph.Graph.from_str(open(profile_filename, 'r').read())

    # Zero out all metadata associated with inputs in graph, since the optimizer
    # shouldn't really get a choice with where to place the input (should always
    # be in the first stage).
    sources = gr.sources()
    nodes_to_remove = OrderedDict()
    # assert(len(sources)) == 1, "only supports linear chains for now ..."
    ip_activation_size = 0
    for source in sources:
        if source.node_desc.startswith("Input"):
            source.forward_compute_time = 0.0
            source.backward_compute_time = 0.0
            ip_activation_size += source.activation_size
            source.activation_size = 0.0
            source.parameter_size = 0.0
            nodes_to_remove[source] = []
            for out_node in gr.edges[source.node_id]:
                nodes_to_remove[source].append(out_node)
            gr.remove_node(source)

    # Remove all unneeded sinks that are not used, makes code generation and
    # optimization easier.
    sinks = gr.sinks()
    for sink in sinks:
        if sink.node_desc.startswith("__getitem__"):
            gr.remove_node(sink)

    antichain_gr = gr.antichain_dag()
    states = antichain_gr.topological_sort()
    unsplittables = set()
    if verbose:
        print("Total number of states: %d" % len(states))
    print("printing antichains")
    for state in states:
        print(state.antichain)
    states_indices = {}
    for i in range(len(states)):
        states_indices[states[i]] = i
    for i in range(len(states)):
        for antichain_node in states[i].antichain:
            states[i].output_activation_size += gr.nodes[antichain_node].activation_size
            print("state {} {} - {}".format(i, antichain_node, gr.nodes[antichain_node].node_desc))
            if "inplace=True" in gr.nodes[antichain_node].node_desc:
                unsplittables.add(i)
    print(unsplittables)
    
    for i in range(len(states)):
        antichain = states[i].antichain
        
        all_predecessors = gr.all_predecessors(antichain)  #contains the current node as well 
        states[i].compute_time = 0.0
        states[i].activation_size = 0.0
        states[i].parameter_size = 0.0
        
        for predecessor in all_predecessors:
            states[i].compute_time += ((predecessor.forward_compute_time +
                                        predecessor.backward_compute_time) / 1000.0) # cumulative finish time
            
            if not ("inplace=True" in predecessor.node_desc): 
                states[i].activation_size += predecessor.activation_size                 # cumulative activation size
            #else:
                #print("detected relu")
            
            #states[i].activation_size += predecessor.activation_size
            states[i].parameter_size += 3*predecessor.parameter_size                   # 2 for BW_Pass also 
    
    gr.reset()
    

    output_activation_sizes = [state.output_activation_size for state in states]
    input_activation_sizes = [ip_activation_size] + output_activation_sizes[:-1]
    all_predecessor_ids = [[states_indices[predecessor] for predecessor in
                            antichain_gr.predecessors(states[i].node_id)]
                           for i in range(len(states))] #this has the actual predecesssors without the node


    compute_times = []
    activation_sizes = []
    parameter_sizes = []
    for i in range(len(states)+1):
        compute_times_row = []
        activation_sizes_row = []
        parameter_sizes_row = []
        for j in range(len(states)):
            if i == 0:
                compute_times_row.append(states[j].compute_time)
                activation_sizes_row.append(states[j].activation_size)
                parameter_sizes_row.append(states[j].parameter_size)
            else:
                if j > (i-1):
                    compute_times_row.append(states[j].compute_time -
                        states[i-1].compute_time)
                    activation_sizes_row.append(states[j].activation_size -
                        states[i-1].activation_size)
                    parameter_sizes_row.append(states[j].parameter_size -
                        states[i-1].parameter_size)
                else:
                    compute_times_row.append(None)
                    activation_sizes_row.append(None)
                    parameter_sizes_row.append(None)
        compute_times.append(compute_times_row)
        activation_sizes.append(activation_sizes_row)
        parameter_sizes.append(parameter_sizes_row)

    compute_times_with_pipeline_limits =[]
    # upper_limit_on_pipeline = np.zeros((len(states)+1, len(states)), dtype=np.int)

    for i in range(len(states)+1):
        row_i = [] 
        for j in range(len(states)):
            col_j = [None]*(pipeline_limit+1)
            if compute_times[i][j]:
                if use_memory_constraint:
                    mem = activation_sizes[i][j] + parameter_sizes[i][j] + \
                        input_activation_sizes[i]
                    max_grad_mem = input_activation_sizes[i] + output_activation_sizes[i]
                    for k in range(i,j):
                        max_grad_mem = max(max_grad_mem, output_activation_sizes[k]+output_activation_sizes[k+1]) 
                    mem += max_grad_mem
                    pl = min(math.floor(memory_size / mem), pipeline_limit)
                else:
                    pl = pipeline_limit
                #upper_limit_on_pipeline[i, j] = pl
                for k in range(1,pipeline_limit+1):
                    if k <= pl:
                        col_j[k] = compute_times[i][j]
            row_i.append(col_j)
        compute_times_with_pipeline_limits.append(row_i)
    
    print(compute_times_with_pipeline_limits[0][0])
    print(compute_times_with_pipeline_limits[0][1])
    print(compute_times_with_pipeline_limits[0][2])
    print(compute_times_with_pipeline_limits[0][5])
    print(compute_times_with_pipeline_limits[0][-1])
    
    counter = 1
    all_As = []
    num_machines_in_machine = 1
    for num_machines, network_bandwidth in zip(all_num_machines, network_bandwidths):
        print("Solving optimization problem with %d machines with inter-machine bandwidth of %.2f GB/s" % (num_machines, network_bandwidth / 10**9))
        
        
        A = compute_partitioning(compute_times_with_pipeline_limits, activation_sizes, parameter_sizes, input_activation_sizes,
                                 output_activation_sizes, all_predecessor_ids,
                                 num_machines, num_machines_in_machine,
                                 network_bandwidth, pipeline_limit, unsplittables,
                                 final_level=(counter==len(network_bandwidths)))
        num_machines_in_machine = num_machines
        for i in range(len(compute_times_with_pipeline_limits)):
            for j in range(len(compute_times_with_pipeline_limits[0])):
                for pl in range(1,pipeline_limit+1):
                    compute_times_with_pipeline_limits[i][j][pl] = A[i][j][-1][pl][0]
        counter += 1
        all_As.append(A)
    
    assert compute_times_with_pipeline_limits[0][-1][pipeline_limit], "Memory Bounds exceeded, try with lower pipeline limit"  
    # exit()

    # with open("temp.txt", "w") as f:
    #     for i in range(len(compute_times_with_pipeline_limits)):
    #         for j in range(len(compute_times_with_pipeline_limits[0])):
    #             if compute_times_with_pipeline_limits[i][j][0]:
    #                 f.write("{0:.6f} ".format(compute_times_with_pipeline_limits[i][j][0]))
    #             else:
    #                 f.write("None ")
    #         f.write("\n")

    
    splits = [(0, len(states))]
    pipeline_limits_of_split = [pipeline_limit]
    i = len(all_As) - 1

    charm_nn_machine_assignments = defaultdict(list)
    while i >= 0:
        print("======================================")
        print("Level %d" % (i+1))
        print("======================================")
        new_splits = []
        new_pipeline_limits = []
        stage_id = 0
        for index, (start, end) in enumerate(splits):
            partial_splits, machines_used = \
                analyze_partitioning(all_As[i], states, start, end,
                                     network_bandwidths[i], all_num_machines[i], pipeline_limits_of_split[index], input_activation_sizes,
                                     activation_compression_ratio,
                                     print_configuration, verbose)
            start_point = start
            machine_id_start = 0 
            
            for k,split in enumerate(partial_splits):
                new_splits.append((start_point, split))
                if i == 0:
                    predecessors = gr.all_predecessors(states[split-1].antichain)
                    for predecessor in predecessors:
                        if predecessor.stage_id is None:
                            predecessor.set_stage_id(stage_id)
                
        
                for layer_id in range(start_point, split):
                    charm_nn_machine_assignments[layer_id].append((machine_id_start,
                                                    machine_id_start+machines_used[k]-1))
                
                start_point = split
                stage_id += 1

                machine_id_start += machines_used[k]
                sub_level_pl = pipeline_limits_of_split[index] // machines_used[k] 
                if pipeline_limits_of_split[index] % machines_used[k] != 0:
                    sub_level_pl += 1
                new_pipeline_limits.append(sub_level_pl)

            new_splits.append((start_point, end))
            for layer_id in range(start_point, end):
                    charm_nn_machine_assignments[layer_id].append((machine_id_start,
                                                    machine_id_start+machines_used[-1]-1))
            sub_level_pl = pipeline_limits_of_split[index] // machines_used[-1]
            if pipeline_limits_of_split[index] % machines_used[-1] != 0:
                    sub_level_pl += 1
            new_pipeline_limits.append(sub_level_pl)
            if i == 0:
                predecessors = gr.all_predecessors(states[end-1].antichain)
                for predecessor in predecessors:
                    if predecessor.stage_id is None:
                        predecessor.set_stage_id(stage_id)
            stage_id += 1
        print("Total number of stages: %d" % stage_id)
        splits = new_splits
        pipeline_limits_of_split = new_pipeline_limits
        i -= 1

    #print(charm_nn_machine_assignments)
    num_per_stage = all_num_machines[::-1][1:] + [1]
    #print(num_per_stage)
    charm_nn_array_map = np.zeros((len(states), pipeline_limit), dtype=np.int)
    for layer, assignments in charm_nn_machine_assignments.items():
        min_pe_id = 0
        max_pe_id = 0
        for i, assgn in enumerate(assignments):
            min_pe_id += assgn[0]*num_per_stage[i]
            max_pe_id += assgn[1]*num_per_stage[i]
        #print("Layer number {} Min PE {} Max PE {}".format(layer, min_pe_id, max_pe_id))
        
        for i in range(pipeline_limit):
            charm_nn_array_map[layer][i] = min_pe_id + (i % (max_pe_id - min_pe_id + 1) )
    
    print(charm_nn_array_map)

    charm_nn_config = {}
    for i,state in enumerate(states):
        assert len(state.antichain) == 1
        node = state.antichain[0]
        node_desc = gr.nodes[node].node_desc
        #print(node_desc)
        charm_nn_config[i] = node_desc
    
    for source in nodes_to_remove:
        for out_node in nodes_to_remove[source]:
            source.stage_id = 0
            gr.add_edge(source, out_node)

    if output_directory is not None:
        total_num_machines = 1
        for num_machines in all_num_machines:
            total_num_machines *= num_machines
        # Commented out because not working
        # gr.to_dot(os.path.join(output_directory, "gpus=%d" % total_num_machines))
        gr_str = str(gr)
        with open(os.path.join(output_directory, "gpus=%d.txt" % total_num_machines), 'w') as f:
            f.write(gr_str)
        with open(os.path.join(output_directory, "charm_nn_array_map.txt"), "w") as f:
            np.savetxt(f,charm_nn_array_map.astype(int))
        with open(os.path.join(output_directory, "charm_nn_config.json"), "w") as f:
            json.dump(charm_nn_config, f, indent=2)

    

    total_time = states[-1].compute_time
    total_parameter_size = states[-1].parameter_size
    data_parallel_total_time = total_time
    num_machines_in_machine = 1
    for (num_machines, network_bandwidth) in zip(all_num_machines, network_bandwidths):
        data_parallel_communication_time = (
            (4 * (num_machines - 1) * total_parameter_size) /
            (network_bandwidth * num_machines)) / num_machines_in_machine
        data_parallel_total_time = sum(
            [data_parallel_total_time, data_parallel_communication_time]) / num_machines
        num_machines_in_machine = num_machines
    pipeline_parallel_total_time = all_As[-1][0][len(states)-1][num_machines-1][pipeline_limit][0]

    if verbose:
        print()
        print("Time taken by single-stage pipeline:", total_time)
        print("Time per stage in pipeline:", pipeline_parallel_total_time)
        print("Throughput increase (compared to single machine):",
              total_time / pipeline_parallel_total_time)
        dp_str = ",".join([str(elem) for elem in all_num_machines])
        print(("[Note that single-machine and (%s)-machine DP might not fit "
               "given memory constraints]") % dp_str)
        print("Throughput increase of (%s)-machine DP compared to single "
              "machine:" % dp_str, total_time / data_parallel_total_time)
        print("Throughput increase (compared to (%s)-machine DP):" % dp_str,
              data_parallel_total_time / pipeline_parallel_total_time)
    return pipeline_parallel_total_time, data_parallel_total_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Run PipeDream's optimizer for replicated settings")
    )
    parser.add_argument('-n', "--all_num_machines", nargs='+', type=int,
                        help="Number of machines available")
    parser.add_argument('-f', "--profile_filename", required=True,
                        help="Profile filename")
    parser.add_argument('-b', "--network_bandwidths", type=float, nargs='+', default=[1000000000],
                        help="Available network bandwidth in bytes/sec")
    parser.add_argument('-s', "--memory_size", type=float, default=16000000000,
                        help="Amount of memory available on each machine")
    parser.add_argument("--straight_pipeline", action='store_true',
                        help="No replication across stages")
    parser.add_argument('-o', "--output_directory", default=None, type=str,
                        help="Output directory to dump processed graph")
    parser.add_argument("--use_memory_constraint", action='store_true',
                        help="Enforce memory constraint per machine")
    parser.add_argument("--use_fewer_machines", action='store_true',
                        help="Use fewer machines, if possible")
    parser.add_argument("--activation_compression_ratio", default=None, type=float,
                        help="Compression ratio for activations")
    parser.add_argument('-pl',"--pipeline_limit", type=int, 
                        help="The upper limit on minibatches in a pipeline")

    args = parser.parse_args()
    args = vars(args)

    all_num_machines = args["all_num_machines"]
    profile_filename = args["profile_filename"]
    network_bandwidths = args["network_bandwidths"]
    assert(len(all_num_machines) == len(network_bandwidths))
    memory_size = args["memory_size"]
    straight_pipeline = args["straight_pipeline"]
    output_directory = args["output_directory"]
    use_memory_constraint = args["use_memory_constraint"]
    use_fewer_machines = args["use_fewer_machines"]
    activation_compression_ratio = args["activation_compression_ratio"]
    pipeline_limit = args["pipeline_limit"]
    main(all_num_machines, profile_filename, network_bandwidths, memory_size,
         straight_pipeline, use_memory_constraint, use_fewer_machines,
         activation_compression_ratio, output_directory, pipeline_limit,
         verbose=True)
