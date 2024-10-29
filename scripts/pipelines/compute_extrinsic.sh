#!/bin/bash
# Combined

./scripts/signatures/compute_corpus_surprisal.sh -t combined -d childes -s 42
./scripts/signatures/compute_corpus_surprisal.sh -t combined -d childes -s 123
./scripts/signatures/compute_corpus_surprisal.sh -t combined -d childes -s 28053

./scripts/signatures/compute_corpus_surprisal.sh -t combined -d babylm -s 42
./scripts/signatures/compute_corpus_surprisal.sh -t combined -d babylm -s 123
./scripts/signatures/compute_corpus_surprisal.sh -t combined -d babylm -s 28053

./scripts/signatures/compute_corpus_surprisal.sh -t combined -d unified -s 42
./scripts/signatures/compute_corpus_surprisal.sh -t combined -d unified -s 123
./scripts/signatures/compute_corpus_surprisal.sh -t combined -d unified -s 28053


# childes

./scripts/signatures/compute_corpus_surprisal.sh -t positive -d childes -s 42
./scripts/signatures/compute_corpus_surprisal.sh -t positive -d childes -s 123
./scripts/signatures/compute_corpus_surprisal.sh -t positive -d childes -s 28053

./scripts/signatures/compute_corpus_surprisal.sh -t negative -d childes -s 42
./scripts/signatures/compute_corpus_surprisal.sh -t negative -d childes -s 123
./scripts/signatures/compute_corpus_surprisal.sh -t negative -d childes -s 28053

# babylm

./scripts/signatures/compute_corpus_surprisal.sh -t positive -d babylm -s 42
./scripts/signatures/compute_corpus_surprisal.sh -t positive -d babylm -s 123
./scripts/signatures/compute_corpus_surprisal.sh -t positive -d babylm -s 28053

./scripts/signatures/compute_corpus_surprisal.sh -t negative -d babylm -s 42
./scripts/signatures/compute_corpus_surprisal.sh -t negative -d babylm -s 123
./scripts/signatures/compute_corpus_surprisal.sh -t negative -d babylm -s 28053

# unified

./scripts/signatures/compute_corpus_surprisal.sh -t positive -d unified -s 42
./scripts/signatures/compute_corpus_surprisal.sh -t positive -d unified -s 123
./scripts/signatures/compute_corpus_surprisal.sh -t positive -d unified -s 28053

./scripts/signatures/compute_corpus_surprisal.sh -t negative -d unified -s 42
./scripts/signatures/compute_corpus_surprisal.sh -t negative -d unified -s 123
./scripts/signatures/compute_corpus_surprisal.sh -t negative -d unified -s 28053

