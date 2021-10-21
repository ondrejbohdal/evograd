# tgt_lang is one of [qu, cdo, ilo, xmf, mhr, mi, tk, gn] - we give an example with qu
# you need to specify your own OUTPUT_DIR

# MetaXL with EvoGrad
python mtrain.py \
--data_dir data/panx_dataset \
--bert_model xlm-roberta-base \
--tgt_lang qu \
--task_name panx \
--train_max_seq_length 200 \
--max_seq_length 512 \
--epochs 20 \
--batch_size 12 \
--method metaxl_evograd \
--output_dir ${OUTPUT_DIR} \
--warmup_proportion 0.1 \
--main_lr 3e-05 \
--meta_lr 1e-06 \
--train_size 5000 \
--target_train_size 100 \
--source_languages en \
--source_language_strategy specified \
--layers 12 \
--struct perceptron \
--tied  \
--transfer_component_add_weights \
--tokenizer_dir None \
--bert_model_type ori \
--bottle_size 384 \
--portion 2 \
--data_seed 42  \
--seed 11 \
--do_train  \
--do_eval 

# MetaXL with the original meta-learning approach
python mtrain.py \
--data_dir data/panx_dataset \
--bert_model xlm-roberta-base \
--tgt_lang qu \
--task_name panx \
--train_max_seq_length 200 \
--max_seq_length 512 \
--epochs 20 \
--batch_size 12 \
--method metaxl \
--output_dir ${OUTPUT_DIR} \
--warmup_proportion 0.1 \
--main_lr 3e-05 \
--meta_lr 1e-06 \
--train_size 5000 \
--target_train_size 100 \
--source_languages en \
--source_language_strategy specified \
--layers 12 \
--struct perceptron \
--tied  \
--transfer_component_add_weights \
--tokenizer_dir None \
--bert_model_type ori \
--bottle_size 384 \
--portion 2 \
--data_seed 42  \
--seed 11 \
--do_train  \
--do_eval 

# Joint-training baseline
python mtrain.py \
--data_dir data/panx_dataset \
--bert_model xlm-roberta-base \
--tgt_lang qu \
--task_name panx \
--train_max_seq_length 200 \
--max_seq_length 512 \
--epochs 20 \
--batch_size 12 \
--method joint_training \
--output_dir ${OUTPUT_DIR} \
--warmup_proportion 0.1 \
--main_lr 3e-05 \
--meta_lr 1e-06 \
--train_size 5000 \
--target_train_size 100 \
--source_languages en \
--source_language_strategy specified \
--layers 12 \
--struct perceptron \
--tied  \
--transfer_component_add_weights \
--tokenizer_dir None \
--bert_model_type ori \
--bottle_size 384 \
--portion 2 \
--data_seed 42  \
--seed 11 \
--do_train  \
--do_eval 