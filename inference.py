# This file is derived from test.py in the PatchCore implementation by
# ComputermindCorp:
# https://github.com/ComputermindCorp/patchcore
#
# The original PatchCore implementation is licensed under the
# Apache License, Version 2.0.
#
# Modifications were made for the ceiling anomaly detection study,
# including inference processing, anomaly-map output, and calculation
# of the anomaly area ratio.
#
# See the LICENSE file in this repository for the applicable license terms.

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import omegaconf
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import numpy as np
import cv2
from sklearn import metrics
import copy
import torch

from models.patch_core import PatchCore
from common.pytorch_custom_dataset import ImagePaths
from common.benchmark import Benchmark
from models.patch_core import visualize

def write_csv(
        save_path: str,
        csv_results: list,
        cfg: omegaconf.dictconfig.DictConfig,
        net: PatchCore,
        th: float,
        encoding: str='shift_jis',
    ):
    """csvファイルを作成・保存

    Args:
        save_path (str): 保存先パス
        csv_results (list): csvに書き込むデータ
        cfg (omegaconf.dictconfig.DictConfig): 設定ファイルオブジェクト
        net (PatchCore): PatchCoreオブジェクト
        th (float): しきい値
        encoding (str, optional): csvファイルのテキストエンコーディング. Defaults to 'shift_jis'.
    """

    with open(save_path, 'w', encoding=encoding) as f:
        writer = csv.writer(f, lineterminator='\n')
        
        # 画像ごとの結果（ヘッダー）
        header = ["No", "image"]

        header.append("Anomaly Score")
#
        header.append("Anomaly Area Ratio")
#
        header.append("Result")

        # 書き込み
        writer.writerow(header)

        # 画像ごとの結果
        writer.writerows(csv_results)

def test(
        cfg: omegaconf.dictconfig.DictConfig,
        visible_bench: bool=False,
    ):
    """テストメイン処理

    Args:
        cfg (omegaconf.dictconfig.DictConfig): 設定情報
        enable_bench (bool, optional): 処理速度ベンチマークの表示有無. Defaults to False.
    """
    # 結果出力パス設定
    if cfg.output_root_path is None or cfg.output_root_path == "":
        output_root_path = None
    else:
        output_root_path = Path(cfg.output_root_path)
        output_root_path.mkdir(exist_ok=True, parents=True)

    # PatchCoreモデル
    net = PatchCore.load_weights(cfg.weights_path, cfg.device)

    # ベンチマーク設定
    net._enable_bench()

    if visible_bench:
        net._show_bench()

    # データローダー
    # dataset
    test_dataset = ImagePaths.create_from_root_paths(
        cfg.test_data_paths,
        label_list=cfg.labels,
        transform = net.get_transform(),
        resize=net.get_resize(),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 初期設定
    results = []
    corrects = np.zeros(len(test_loader), np.int32)
    preds = np.zeros(len(test_loader), np.int32)


    csv_results = []

    # 結果ファイルパス設定
    if output_root_path is None:
        output_heatmap_root_path = None
        csv_save_path = None
    else:
        # 結果CSVファイル
        csv_save_path = output_root_path / f"{Path(cfg.weights_path).stem}_ano.csv"

        # 結果画像ファイル
        output_heatmap_root_path = output_root_path / f"heatmap_{Path(cfg.weights_path).stem}_ano"
        output_heatmap_root_path.mkdir(parents=True, exist_ok=True)

        ng_output_path = output_heatmap_root_path / "Damage"
        ng_output_path.mkdir(exist_ok=True, parents=True)
        ok_output_path = output_heatmap_root_path / "No_Damage"
        ok_output_path.mkdir(exist_ok=True, parents=True)

    # しきい値
    th = 0.5 if cfg.th is None else cfg.th
    anomaly_ratio = 0

    # テスト
    for i, (x, label, paths) in enumerate(test_loader):
        anomaly_score, anomaly_map_org, pred = net.predict(x, th=th)
        anomaly_map = copy.deepcopy(anomaly_map_org)
        score_value = torch.linalg.norm(anomaly_map)
        anomaly_map_m = anomaly_map.squeeze()
        anomaly_ratio = 0
        for ii in range(0,224):
            for jj in range(0,224):
                if anomaly_map_m[ii][jj] < th:
                    anomaly_map_m[ii][jj] = 0
                else:
                    anomaly_ratio = anomaly_ratio + 1

        score_value_m = torch.linalg.norm(anomaly_map_m)
        anomaly_ratio = anomaly_ratio / (224 * 224)

        label = label.tolist()


        # 結果出力
        if output_root_path is not None:
            # csv
            csv_result = [i+1, Path(paths[0]).name]

            csv_result.append(anomaly_score.item())
            csv_result.append(anomaly_ratio)

            result = (anomaly_score.item() <= th)
            csv_result.append("No Damage" if result else "Damage")
            csv_results.append(csv_result)

            # heatmap画像生成・保存
            if output_heatmap_root_path is not None:
                heatmap_save_path = output_heatmap_root_path / f"{Path(paths[0]).stem}_heatmap.png"
                heatmap_add_save_path = output_heatmap_root_path / f"{Path(paths[0]).stem}.png"
                heatmap_color_bar = output_heatmap_root_path / "color_bar.png"

                im_org = cv2.imread(paths[0])

                im_heatmap = visualize.create_heatmap_image(anomaly_map, org_size=im_org.shape)
                #print(np.linalg.norm(anomaly_map))
                #print(torch.norm(anomaly_map))

                im_add = visualize.add_image(im_heatmap, im_org, alpha=0.5)

                visualize.create_color_bar_image(save_path=str(heatmap_color_bar))

                if result == False:                    
                    print("NG")
                    #score_value = torch.linalg.norm(anomaly_map)
                    #print(anomaly_map_m.shape)
                    im_ng_heatmap = visualize.create_heatmap_image(anomaly_map, org_size=im_org.shape)
                    im_ng_add = visualize.add_image(im_heatmap, im_org, alpha=0.5)
                    cv2.imwrite(str(ng_output_path / heatmap_save_path.name), im_ng_heatmap)
                    cv2.imwrite(str(ng_output_path / heatmap_add_save_path.name), im_ng_add)
                else:
                    print("OK")
                    score_value = torch.linalg.norm(anomaly_map)
                    #print(anomaly_map_m.shape)
                    im_ok_heatmap = visualize.create_heatmap_image(anomaly_map, org_size=im_org.shape)
                    im_ok_add = visualize.add_image(im_heatmap, im_org, alpha=0.5)
                    cv2.imwrite(str(ok_output_path / heatmap_save_path.name), im_ok_heatmap)
                    cv2.imwrite(str(ok_output_path / heatmap_add_save_path.name), im_ok_add)


    # CSVファイル書き込み
    if csv_save_path is not None:
        write_csv(
            csv_save_path,
            csv_results,
            cfg,
            net,
            th,
        )
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='config path')
    parser.add_argument('--show-bench', action='store_true', help='enabel benchmark')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    test(cfg, args.show_bench)
