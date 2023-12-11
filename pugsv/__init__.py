import sys
import os
import logging
import numpy as np
from time import strftime, localtime

import datetime
import shutil
import pysam

import argparse
import multiprocessing
import traceback

def parse_arguments(arguments = sys.argv[1:]):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="""pugSV \n \nShort Usage: pugSV [parameters] -o <output path> -b <input bam path> -g <reference> -m <model path>""")


    required_params = parser.add_argument_group("Input/Output parameters")
    required_params.add_argument('-o', dest="out_path", type=os.path.abspath, required=True, help='Absolute path to output ')
    required_params.add_argument('-b', dest='bam_path', type=os.path.abspath, required=True, help='Absolute path to bam file')
    required_params.add_argument('-g', dest='genome', type=os.path.abspath, required=True, help='Absolute path to your reference genome (.fai required in the directory)')

    optional_params = parser.add_argument_group("Optional parameters")
    optional_params.add_argument('-c', dest="chrom", type=str, default=None, help='Specific region (chr1:xxx-xxx) or chromosome (chr1) to detect')

    optional_params.add_argument('--debug', action="store_true", default=False,
                                 help='Activate debug mode and keep intermedia outputs (default: %(default)s)')

    collect_params = parser.add_argument_group("Collect parameters")

    collect_params.add_argument("--min_mapq", type=int, default=10, help='Minimum mapping quality of reads to consider (default: %(default)s)')
    collect_params.add_argument("--min_sv_size", type=int, default=50, help='Minimum SV size to detect (default: %(default)s)')
    collect_params.add_argument("--max_sv_size", type=int, default=1000000, help='Maximum SV size to detect (default: %(default)s)')
    collect_params.add_argument("--window_size", type=int, default=10000000, help='The sliding window size in segment collection (default: %(default)s)')

    options = parser.parse_args(arguments)

    return options

if __name__ == '__main__':
    options = parse_arguments()

    work_dir = options.out_path
    if not os.path.exists(work_dir):
        print(f'Create the output directory {work_dir}')
        os.mkdir(work_dir)
        
    log_format = logging.Formatter("%(asctime)s [%(levelname)-7.7s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)


    fileHandler = logging.FileHandler("{0}/SVision_{1}.log".format(work_dir, strftime("%y%m%d_%H%M%S", localtime())), mode="w")
    fileHandler.setFormatter(log_format)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(log_format)
    root_logger.addHandler(fileHandler)
    ## End ADD

    sample_path = options.bam_path
    aln_file = pysam.AlignmentFile(sample_path)

    ## Check the input
    logging.info("INPUT BAM: {0}".format(os.path.abspath(sample_path)))
    try:
        if aln_file.header["HD"]["SO"] == "coordinate":
            try:
                aln_file.check_index()
            except ValueError:
                logging.warning("Input BAM file is missing a valid index. Please generate with 'samtools faidx'. Continuing without genotyping for now..")
                options.skip_genotyping = True
            except AttributeError:
                logging.warning("pysam's .check_index raised an Attribute error. Something is wrong with the input BAM file.")
                exit()
    except:
        logging.error("This is not a coordinate sorted BAM file")
        exit()

    logging.info('******************** Start pugSV********************')
    logging.info("CMD: {0}".format(" ".join(sys.argv)))
    logging.info("WORKDIR DIR: {0}".format(os.path.abspath(work_dir)))
    
    window_size = options.window_size

    if options.contig:
        options.min_support = 1

    task_list_bychrom = {}
    ref_info = aln_file.get_index_statistics()

    all_possible_chrs = pysam.FastaFile(options.genome).references
    logging.info("INPUT GENOME: {0}".format(os.path.abspath(options.genome)))
    if options.chrom == None:

        ## V1.3.6 add for multiprocessing
        for ele in ref_info:
            chrom = ele[0]
            local_ref_len = aln_file.get_reference_length(chrom)

            if chrom not in all_possible_chrs:
                continue

            if options.contig:
                window_size = local_ref_len

            if local_ref_len < window_size:
                if chrom in task_list_bychrom:
                    task_list_bychrom[chrom].append([0, local_ref_len])
                else:
                    task_list_bychrom[chrom] = [[0, local_ref_len]]
            else:
                pos = 0
                round_task_num = int(local_ref_len / window_size)
                for j in range(round_task_num):
                    if chrom in task_list_bychrom:
                        task_list_bychrom[chrom].append([pos, pos + window_size])
                    else:
                        task_list_bychrom[chrom] = [[pos, pos + window_size]]
                    pos += window_size

                if pos < local_ref_len:
                    if chrom in task_list_bychrom:
                        task_list_bychrom[chrom].append([pos, local_ref_len])
                    else:
                        task_list_bychrom[chrom] = [[pos, local_ref_len]]

        ## End add

    else:
        chrom = options.chrom

        ## V1.3.6 added for process a given chrom of region
        if chrom in all_possible_chrs:
            start = 0
            end = aln_file.get_reference_length(chrom)

        else:
            cords = chrom.split(':')[1]
            chrom, start, end = chrom.split(':')[0], int(cords.split('-')[0]), int(cords.split('-')[1])

        task_list_bychrom[chrom] = []

        region_length = end - start + 1

        if region_length < window_size:
            task_list_bychrom[chrom].append([start, end])

        else:
            pos = 0
            round_task_num = int(region_length / window_size)
            for j in range(round_task_num):
                task_list_bychrom[chrom].append([pos, pos + window_size])
                pos += window_size

            if pos < region_length:
                task_list_bychrom[chrom].append([pos, region_length])

        ## END add.

    ## v1.3.6 added for handling task errors
    if len(task_list_bychrom) == 0:
        # print('[ERROR]: No mapped reads in this bam. Exit!')
        logging.error("No mapped reads in the BAM, please check your reference input!")
        exit()
        
    chrom_split_files = {}
    process_pool = multiprocessing.Pool(processes=options.thread_num)
    pool_rets = []

    for chrom, task_list in task_list_bychrom.items():
        part_num = 0
        if chrom not in chrom_split_files.keys():
            chrom_split_files[chrom] = ""

        for task in task_list:
            task_start, task_end = task[0], task[1]

            # pool_rets.append([process_pool.apply_async(run_collection.run_detect,
                                                    #    (options, sample_path, chrom, part_num, task_start, task_end)), chrom, task_start, task_end])

            # run_collection.run_detect(options, sample_path, chrom, part_num, task_start, task_end)

            part_num += 1

    process_pool.close()
    process_pool.join()
    