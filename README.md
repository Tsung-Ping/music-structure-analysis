# music-structure-analysis
A deep learning method for music structure analysis (MSA) that aims to split a music signal into temporal segments and assign a function label (e.g., intro, verse, or chorus) to each of these segments. More details can be found in the paper: Tsung-Ping Chen and Kazuyoshi Yoshii, "Learning Multifaceted Self-Similarity over Time and Frequency for Music Structure Analysis," International Society of Music Information Retrieval Conference (ISMIR), November 2024.

#
This repository implements the proposed model (Section 3) and the baseline methods (Section 4.4):

- FunctionalSegmentModel: the proposed method
- FunctionalSegmentModel_stAttn: baseline method (ST-MHSA)
- FunctionalSegmentModel_tAttn: baseline method (TE-Only)
- FunctionalSegmentModel_sAttn: baseline method (SE-Only)

