python train.py \
--train_A  \
--train_B  \
--train_C  \
--valid_A  \
--valid_B  \
--valid_C  \
--batch_size \
--aug \
--model_type re3 \
--optimizer Adam \
--loss_type MSP_CE \
--epoch_num 350 \
--class_num 2 \
--backbone_type combine_resnet_18\
--reunit_type  \
--reunit_skipconnec  \
--dropout_probs 0 \
--recurrent_time 0 \
--loss_weights 1 1 1 1 1 1 \
--cross_validate \
--cross_interval 100 \
--lr_manage_type stage \
--epoch_stages 0 250 \
--lr_stages 1 0.1 \
--max_lr 0.0001 \
--outlayer_type 12 \
--best_start 300 \
--CE_Thre 0 0 0 0 0 0  \
--edge_mask_initial 0.2 0 0 0 0 0 \
--save_dir ./