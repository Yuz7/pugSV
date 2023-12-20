import os
from collections import defaultdict
from enum import Enum
from pysam import VariantFile
import random
import bisect
from intervaltree import IntervalTree

BED_FILE_TYPE = Enum("BED_FILE_TYPE", 'BED, BEDPE')

class GenomeInterval(tuple):
    def __new__(cls, chr_name, start, end):
        return tuple.__new__(GenomeInterval, (chr_name, start, end))

    def __init__(self, chr_name, start, end):
        self.chr_name = chr_name
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start

    def mean(self):
        return self.start + int((self.end - self.start)/2)

    def pad(self, chr_index, padding, rand=False):
        start_pad = padding
        end_pad = padding
        if rand:
            split = random.randint(0, padding//2)
            start_pad = padding//2 + split
            end_pad = 2*padding - start_pad
        start = max(0, self.start - start_pad)
        end = min(self.end + end_pad, chr_index.chr(chr_index.tid(self.chr_name)).len)
        return GenomeInterval(self.chr_name, start, end)

    def __str__(self):
        return "%s_%d-%d" % (self.chr_name, self.start, self.end)

    def __lt__(self, interval):
        return self.start < interval.start

    def to_list(self, chr_index):
        return [chr_index.tid(self.chr_name), self.start, self.end]

    @staticmethod
    def from_list(interval_list, chr_index):
        return GenomeInterval(chr_index.chr(interval_list[0]).name, interval_list[1], interval_list[2])

    @staticmethod
    def to_interval(chr_name, pos):
        return GenomeInterval(chr_name, pos, pos + 1)

class SVIntervalTree:
    def __init__(self, intervals):
        self.chr2tree = defaultdict(IntervalTree)
        for interval in intervals:
            self.add(interval)

    # def overlaps(self, interval, delta=50, frac=0.2):
    #     start = max(0, interval.intervalA.start - delta)
    #     end = interval.intervalB.start + delta
    #     if not self.chr2tree[interval.intervalA.chr_name].overlaps(start, end):
    #         return False
    #     candidates = self.chr2tree[interval.intervalA.chr_name].overlap(start, end)
    #     for c in candidates:
    #         candidate_interval = c.data
    #         overlap_start = max(candidate_interval.intervalA.start, start)
    #         overlap_end = min(candidate_interval.intervalB.start, end)
    #         candidate_len = candidate_interval.intervalB.start - candidate_interval.intervalA.start
    #         if overlap_start < overlap_end:
    #             if float((overlap_end - overlap_start) / min(end - start, candidate_len)) >= frac:
    #                 return True
    #     return False

    def add(self, interval):
        self.chr2tree[interval.intervalA.chr_name].addi(interval.intervalA.start,
                                                        interval.intervalA.end, interval)

class BedRecord:
    def __init__(self, sv_type, intervalA, intervalB=None, aux=None):
        self.intervalA = intervalA
        self.intervalB = intervalB
        self.sv_type = sv_type
        self.aux = aux
        self.format = BED_FILE_TYPE.BEDPE if intervalB is not None else BED_FILE_TYPE.BED

    @staticmethod
    def parse_bed_line(line, bed_file_type=BED_FILE_TYPE.BEDPE):
        fields = line.strip().split()  # "\t")
        assert len(fields) >= 3, "Unexpected number of fields in BED: %s" % line
        chr_name, start, end = fields[:3]
        intervalA = GenomeInterval(chr_name, int(start), int(end))
        # intervalB = None
        # if bed_file_type == BED_FILE_TYPE.BEDPE:
        #     assert len(fields) >= 6, "Unexpected number of fields in BEDPE: %s" % line
        #     chrB, startB, endB = fields[3:6]
        #     intervalB = GenomeInterval(chrB, int(startB), int(endB))
        req_fields = 3 if bed_file_type == BED_FILE_TYPE.BED else 6
        name = fields[req_fields] if len(fields) > req_fields else 'NA'
        # aux = {'score': fields[req_fields + 1] if len(fields) > req_fields + 1 else 0,
        #        'zygosity': constants.ZYGOSITY_ENCODING_BED[fields[req_fields + 2]] if len(fields) > req_fields + 2
        #        else constants.ZYGOSITY.UNK}  # TODO: strand vs zygosity
        return BedRecord(name, intervalA)

    def get_sv_type(self, to_vcf_format=False):
        if to_vcf_format:
            return "<%s>" % self.sv_type
        return self.sv_type

    def get_score(self):
        return self.aux['score']

    # def get_zygosity(self):
    #     return self.aux['zygosity']

    # def get_sv_type_with_zyg(self):
    #     return "%s-%s" % (self.sv_type, self.get_zygosity().value)

    # @staticmethod
    # def parse_sv_type_with_zyg(sv_type):
    #     if "-" in sv_type:
    #         sv_type, zyg = sv_type.split("-")
    #         return sv_type, constants.ZYGOSITY(zyg)
    #     else:
    #         return sv_type, None

    def __str__(self):
        return "%s, %s, %s" % (self.sv_type, str(self.intervalA), str(self.intervalB))

    def get_name(self):
        return "%s_%s_%s" % (self.sv_type, str(self.intervalA), str(self.intervalB))

    def to_bedpe(self):
        assert self.format == BED_FILE_TYPE.BEDPE
        return "%s\t%s\t%s\t%s\t%s\t%s\t%s" % (self.intervalA.chr_name, self.intervalA.start, self.intervalA.end,
                                               self.intervalB.chr_name, self.intervalB.start, self.intervalB.end,
                                               self.sv_type)

    # def to_bedpe_aux(self):
    #     assert self.format == BED_FILE_TYPE.BEDPE
    #     return "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (self.intervalA.chr_name,
    #                                                    self.intervalA.start, self.intervalA.end,
    #                                                    self.intervalB.chr_name,
    #                                                    self.intervalB.start, self.intervalB.end,
    #                                                    self.sv_type, self.aux['score'],
    #                                                    constants.ZYGOSITY_GT_BED[self.aux['zygosity']])
    
    def to_bed(self):
        assert self.format == BED_FILE_TYPE.BED
        return "%s\t%s\t%s\t%s" % (self.intervalA.chr_name, self.intervalA.start, self.intervalA.end, self.sv_type)

    @staticmethod
    def get_bedpe_header():
        return '#chrom1\tstart1\tstop1\tchrom2\tstart2\tstop2\tname'

    @staticmethod
    def get_bedpe12_header():
        return '#chrom1\tstart1\tstop1\tchrom2\tstart2\tstop2\tname\tscore\tstrand1\tstrand2\tfilter\tinfo'

    @staticmethod
    def get_bedpe_aux_header():
        return '#chrom1\tstart1\tstop1\tchrom2\tstart2\tstop2\tname\tscore\tgt'

    @staticmethod
    def compare(rec1, rec2):
        return rec1.intervalA.start - rec2.intervalA.start

    @staticmethod
    def compare_by_score(rec1, rec2):
        return rec1.get_score() - rec2.get_score()

    def __lt__(self, rec):
        return self.intervalA.__lt__(rec.intervalA)

def bed_iter(bed_fname, bed_file_type=BED_FILE_TYPE.BEDPE, keep_chrs=None, exclude_names=None):
    with open(bed_fname, 'r') as bed_file:
        for line in bed_file:
            if line.startswith('#') or line.isspace():
                continue
            record = BedRecord.parse_bed_line(line, bed_file_type)
            if exclude_names is not None and record.sv_type in exclude_names:
                continue
            if keep_chrs is None or (record.intervalA.chr_name in keep_chrs and
                                     (record.intervalB is None or record.intervalB.chr_name in keep_chrs)):
                yield record

def vcf_iter(vcf_fname, min_size = 30, contig = None, include_types = None):
    if contig is not None:
        vcf_file = VariantFile(vcf_fname)
    else:    
        vcf_file = VariantFile(vcf_fname)
    for rec in vcf_file.fetch():
        if 'SVTYPE' not in rec.info:
            continue
        sv_type = rec.info['SVTYPE']
        # filter by type
        if include_types is not None and sv_type not in include_types:
            continue
        if 'SVLEN' in rec.info:
            if isinstance(rec.info['SVLEN'], tuple):
                sv_len = int(rec.info['SVLEN'][0])
            else:
                sv_len = int(rec.info['SVLEN'])
        else:
            sv_len = rec.stop - rec.pos
        sv_len = abs(sv_len)
        # filter by length
        if sv_len < min_size:
            continue
        start = int(rec.pos) - 1  # 0-based
        end = start + sv_len
        interval = GenomeInterval(rec.contig, start, end)
        # intervalB = GenomeInterval(rec.contig, end, end + 1)
        if 'GT' in rec.samples[rec.samples[0].name]:
            gt = rec.samples[rec.samples[0].name]['GT']
        else:
            gt = (None, None)
        if gt[0] == 0 and gt[1] == 0:
            # print("!Found a HOM ref entry in the VCF: ", rec)
            pass
        # zygosity = constants.ZYGOSITY_ENCODING[gt] if gt[0] is not None else constants.ZYGOSITY.UNK
        # aux = {'score': rec.qual,
            #    'zygosity': zygosity}
        bedpe_record = BedRecord(sv_type, interval)
        yield bedpe_record
                

class BedRecordContainer:
    def __init__(self, fname, contig = None):
        self.ground_truth = defaultdict(IntervalTree)
        iterator = None
        # if 'bed' in fname:
            # iterator = bed_iter(fname, bed_file_type=BED_FILE_TYPE.BED)
        # elif 'vcf' in fname: 
        iterator = vcf_iter(fname, contig=contig)
        assert iterator is not None
        for _, record in enumerate(iterator):
            self.ground_truth[record.intervalA.chr_name].addi(record.intervalA.start, record.intervalA.end, record.sv_type)

    def get_sv_type(self, chrom, start, end):
        if not self.ground_truth[chrom].overlap(start, end):
            return 0
        sv_types = self.ground_truth[chrom].overlap(start, end)
        for sv_type in sv_types:
            return 1 if sv_type == "DEL" else 2
        pass
        
    # def overlap(self, interval):
    #     starts = self.coords_in_interval(interval, self.chr2starts[interval.chr_name])
    #     ends = self.coords_in_interval(interval, self.chr2ends[interval.chr_name])
    #     # remove dups (some records can start and end in this interval)
    #     records = set()
    #     for _, i in starts:
    #         records.add(self.chr2rec[interval.chr_name][i])
    #     for _, i in ends:
    #         records.add(self.chr2rec[interval.chr_name][i])
    #     return list(records)

    def __iter__(self):
        for rec in self.ground_truth:
            yield rec