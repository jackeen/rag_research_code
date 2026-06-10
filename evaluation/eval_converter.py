import pandas as pd

from tools.data_loader import (
    get_csv_log_dir,
    get_no_tail_csv_log_path,
)


def extract_n_and_tag(file_stem: str) -> tuple[int, bool] | None:
    x = file_stem.split("_")
    n = int(x[5])
    if len(x) == 7:
        return (n, True)
    if len(x) == 6:
        return (n, False)
    return None


def eval_collect(sub_path: str) -> list[tuple[int, bool, str]]:
    eval_path = get_csv_log_dir(sub_path)
    target_files = []
    for file in sorted(eval_path.rglob("*")):
        if file.is_file():
            nt = extract_n_and_tag(file.stem)
            if nt is None:
                continue
            else:
                target_files.append((nt[0], nt[1], file.stem))
    return target_files


def avg_eval_ret(sub_path: str, target_files: list[tuple[int, bool, str]]):
    groups: list[list[tuple[int, bool, str]]] = []
    for i in range(0, len(target_files), 3):
        groups.append(target_files[i : i + 3])

    ret_pds = []

    for group_eval in groups:
        file_1 = group_eval[0]
        file_2 = group_eval[1]
        file_3 = group_eval[2]

        material = file_1[0] + 1
        filter_sign = "N"
        if file_1[1]:
            filter_sign = "Y"

        pd_1 = pd.read_csv(get_no_tail_csv_log_path(f"{sub_path}/{file_1[2]}"))
        pd_2 = pd.read_csv(get_no_tail_csv_log_path(f"{sub_path}/{file_2[2]}"))
        pd_3 = pd.read_csv(get_no_tail_csv_log_path(f"{sub_path}/{file_3[2]}"))

        avg_similarity = (
            pd_1["similarity"] + pd_2["similarity"] + pd_3["similarity"]
        ) / 3
        avg_precision = (
            pd_1["context_precision"]
            + pd_2["context_precision"]
            + pd_3["context_precision"]
        ) / 3
        avg_recall = (
            pd_1["context_recall"] + pd_2["context_recall"] + pd_3["context_recall"]
        ) / 3
        avg_faithfulness = (
            pd_1["faithfulness"] + pd_2["faithfulness"] + pd_3["faithfulness"]
        ) / 3

        material_list = [material for _ in avg_similarity]
        filter_sign_list = [filter_sign for _ in avg_similarity]

        group_ret_df = pd.DataFrame(
            {
                "material": material_list,
                "filter": filter_sign_list,
                "question": pd_1["question"].to_list(),
                "cosine_similarity": avg_similarity.to_list(),
                "faithfulness": avg_faithfulness.to_list(),
                "context_precision": avg_precision.to_list(),
                "context_recall": avg_recall.to_list(),
            }
        )

        ret_pds.append(group_ret_df)

    ret = pd.concat(ret_pds, ignore_index=True)
    ret.to_csv(
        get_no_tail_csv_log_path("mutilayer_eval_avg_all"),
        index=False,
        encoding="utf-8",
    )


if __name__ == "__main__":
    sub_path = "z_eval_mutilayer"
    target_files = eval_collect(sub_path)
    avg_eval_ret(sub_path, target_files)
