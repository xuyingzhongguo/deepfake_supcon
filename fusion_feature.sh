python fusion_last_features.py --batch_size 64 --train_list data_lists/f2f_fs_nt_train_000599_frame30.txt --test_list data_lists/f2f_fs_nt_val_600799_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_f2f_fs_nt/ckpt_epoch_20.pth --ckpt_xc save/Xception/xc_f2f_fs_nt_c23 --model_name feature_f2f_fs_nt






#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/deepfakes_test_800999_frame30.txt  --ckpt_supcon save/SupCon/deepfakes_models/whole_model_f2f_fs_nt/ckpt_epoch_20.pth --ckpt_xc save/Xception/xc_f2f_fs_nt_c23 --save_name score_f2f_fs_nt2df
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/face2face_test_800999_frame30.txt  --ckpt_supcon save/SupCon/deepfakes_models/whole_model_f2f_fs_nt/ckpt_epoch_20.pth --ckpt_xc save/Xception/xc_f2f_fs_nt_c23 --save_name score_f2f_fs_nt2f2f
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/faceswap_test_800999_frame30.txt  --ckpt_supcon save/SupCon/deepfakes_models/whole_model_f2f_fs_nt/ckpt_epoch_20.pth --ckpt_xc save/Xception/xc_f2f_fs_nt_c23 --save_name score_f2f_fs_nt2fs
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/neuraltextures_test_800999_frame30.txt  --ckpt_supcon save/SupCon/deepfakes_models/whole_model_f2f_fs_nt/ckpt_epoch_20.pth --ckpt_xc save/Xception/xc_f2f_fs_nt_c23 --save_name score_f2f_fs_nt2nt
#
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/deepfakes_test_800999_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_df_fs_nt/ckpt_epoch_20.pth --ckpt_xc save/Xception/xc_df_fs_nt_c23 --save_name score_df_fs_nt2df
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/face2face_test_800999_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_df_fs_nt/ckpt_epoch_20.pth --ckpt_xc save/Xception/xc_df_fs_nt_c23 --save_name score_df_fs_nt2f2f
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/faceswap_test_800999_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_df_fs_nt/ckpt_epoch_20.pth --ckpt_xc save/Xception/xc_df_fs_nt_c23 --save_name score_df_fs_nt2fs
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/neuraltextures_test_800999_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_df_fs_nt/ckpt_epoch_20.pth --ckpt_xc save/Xception/xc_df_fs_nt_c23 --save_name score_df_fs_nt2nt
#
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/deepfakes_test_800999_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_df_f2f_nt/ckpt_epoch_30.pth --ckpt_xc save/Xception/xc_df_f2f_nt_c23 --save_name score_df_f2f_nt2df
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/face2face_test_800999_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_df_f2f_nt/ckpt_epoch_30.pth --ckpt_xc save/Xception/xc_df_f2f_nt_c23 --save_name score_df_f2f_nt2f2f
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/faceswap_test_800999_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_df_f2f_nt/ckpt_epoch_30.pth --ckpt_xc save/Xception/xc_df_f2f_nt_c23 --save_name score_df_f2f_nt2fs
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/neuraltextures_test_800999_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_df_f2f_nt/ckpt_epoch_30.pth --ckpt_xc save/Xception/xc_df_f2f_nt_c23 --save_name score_df_f2f_nt2nt
#
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/deepfakes_test_800999_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_df_f2f_fs/ckpt_epoch_9.pth --ckpt_xc save/Xception/xc_df_f2f_fs_c23 --save_name score_df_f2f_fs2df
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/face2face_test_800999_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_df_f2f_fs/ckpt_epoch_9.pth --ckpt_xc save/Xception/xc_df_f2f_fs_c23 --save_name score_df_f2f_fs2f2f
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/faceswap_test_800999_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_df_f2f_fs/ckpt_epoch_9.pth --ckpt_xc save/Xception/xc_df_f2f_fs_c23 --save_name score_df_f2f_fs2fs
#python fusion_final_score.py --batch_size 64 --class_number 4 --test_list data_lists/neuraltextures_test_800999_frame30.txt --ckpt_supcon save/SupCon/deepfakes_models/whole_model_df_f2f_fs/ckpt_epoch_9.pth --ckpt_xc save/Xception/xc_df_f2f_fs_c23 --save_name score_df_f2f_fs2nt