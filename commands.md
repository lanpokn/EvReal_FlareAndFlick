python tools/Davis_to_npy.py "E:\2025\event_flick_flare\datasets\DAVIS\test\full_sixFlare-2025_06_25_14_50_08.aedat4" --max_duration 1       

python eval.py -m E2VID FireNet E2VID+ FireNet+ SPADE-E2VID SSL-E2VID ET-Net HyperE2VID -c std -d ECD -qm mse ssim lpips  

python eval.py -m E2VID FireNet E2VID+ FireNet+ SPADE-E2VID SSL-E2VID ET-Net HyperE2VID -c color -d CED -qm mse ssim lpips


MSE,LPIPS:Lower is better, SSIM: higher is better
MSE best:Firenet SSIM best E2VID+ LPIPS best FireNet
               DAVIS a single sequence
Method              MSE    SSIM    LPIPS
-----------  ----------  ------  -------
E2VID             0.395   0.066    0.801
FireNet           0.325   0.089    0.783
E2VID+            0.401   0.208    0.788
FireNet+          0.495   0.110    0.865
SPADE-E2VID       0.476   0.151    1.008
SSL-E2VID         0.496   0.092    1.002
ET-Net            0.470   0.146    0.949
HyperE2VID        0.497   0.052    0.978