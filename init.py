import pandas as pd
from pybiomart import Server
import pyranges as pr

df = pd.read_csv("data/HM450.hg38.manifest.gencode.v36.probeMap", sep='\t')

df = df.rename(columns={
    '#id': 'IlmnID',
    'gene': 'Annot_Gene',
    'chrom': 'Chromosome',
    'chromStart': 'Start',
    'chromEnd': 'End',
    'strand': 'Strand'
})

server = Server(host='http://www.ensembl.org')
dataset = server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl']

gene_annot = dataset.query(attributes=[
    'ensembl_gene_id',
    'chromosome_name',
    'start_position',
    'end_position',
    'strand',
    'external_gene_name'
])

gene_annot = gene_annot[gene_annot['Chromosome/scaffold name'].isin([str(i) for i in range(1,23)] + ['X', 'Y'])]
gene_annot['Chromosome'] = "chr" + gene_annot['Chromosome/scaffold name']

gene_annot = gene_annot.rename(columns={
    'Gene stable ID': 'Ensembl_Gene_ID',
    'Gene start (bp)': 'Start',
    'Gene end (bp)': 'End',
    'Strand': 'Strand',
    'Gene name': 'Gene_Symbol'
})

gene_annot['Strand'] = gene_annot['Strand'].replace({1: '+', -1: '-'})

gr_probes = pr.PyRanges(df[['Chromosome', 'Start', 'End', 'IlmnID', 'Strand']])
gr_genes = pr.PyRanges(gene_annot[['Chromosome', 'Start', 'End', 'Ensembl_Gene_ID', 'Gene_Symbol', 'Strand', 'Gene_Symbol']])

overlap = gr_probes.join(gr_genes, strandedness="same", report_overlap = True)


result_df = overlap.as_df()

result_df = result_df.merge(df[['IlmnID', 'Annot_Gene']], on='IlmnID', how='left')

result_df.to_csv("data/cg_to_ensembl.csv", index=False)