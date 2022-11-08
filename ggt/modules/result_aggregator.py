import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import scipy.stats
from schwimmbad import MultiPool
import click
from scipy.special import expit


def calculate_cdf(data, n=1000, warnings=True):

    # KDE
    kde = scipy.stats.gaussian_kde(data)

    x_domain = np.linspace(min(data), max(data), n)

    y_pdf = kde.pdf(x_domain)

    mode_arg = np.argmax(y_pdf)
    mode = x_domain[mode_arg]

    # Code from StacoOverflow #52221829
    cdf_method = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))
    cdf = cdf_method(x_domain)

    # cdf suitabilitly checking
    # If the end points of the CDF
    # are not 0 and 1, we make it such
    # artificially.
    if cdf[0] != 0:
        if warnings is True and (cdf[0] - 0 > 0.1):
            print("Warning signigicant jump in the lower CDF")
        cdf[0] = 0

    if cdf[len(cdf) - 1] != 1:
        if warnings is True and (1.0 - cdf[len(cdf) - 1] > 0.1):
            print("Warning signigicant jump in the upper CDF")
        cdf[len(cdf) - 1] = 1

    return x_domain, cdf, mode, y_pdf, mode_arg


# from StackOverflow #2566412
def find_idx_nearest_val(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def calculate_ci(data, n=1000, warnings=False):

    x_domain, cdf, mode, y_pdf, mode_arg = calculate_cdf(
        data, n=n, warnings=warnings
    )

    max_arg = len(x_domain) - 1
    min_arg = 0

    upper_arg_arr = []
    lower_arg_arr = []
    percentile_arr = []

    upper_arg = mode_arg
    lower_arg = mode_arg

    # In this loop below, I loop through the x_domain and
    # calculate the different cdf for different values of
    # x
    while (upper_arg < max_arg) or (lower_arg > min_arg):

        if upper_arg < max_arg:
            upper_arg = upper_arg + 1

        if lower_arg > min_arg:
            lower_arg = lower_arg - 1

        upper_arg_arr.append(upper_arg)
        lower_arg_arr.append(lower_arg)
        percentile_arr.append(cdf[upper_arg] - cdf[lower_arg])

    sig_ci_idx = find_idx_nearest_val(percentile_arr, 0.6827)
    sig_ci = tuple(
        [
            x_domain[lower_arg_arr[sig_ci_idx]],
            x_domain[upper_arg_arr[sig_ci_idx]],
        ]
    )

    twosig_ci_idx = find_idx_nearest_val(percentile_arr, 0.9545)
    twosig_ci = tuple(
        [
            x_domain[lower_arg_arr[twosig_ci_idx]],
            x_domain[upper_arg_arr[twosig_ci_idx]],
        ]
    )

    threesig_ci_idx = find_idx_nearest_val(percentile_arr, 0.9973)
    threesig_ci = tuple(
        [
            x_domain[lower_arg_arr[threesig_ci_idx]],
            x_domain[upper_arg_arr[threesig_ci_idx]],
        ]
    )

    return mode, sig_ci, twosig_ci, threesig_ci, x_domain, y_pdf


def expit_custom(x_input, scaling_array=None):

    """Performs a custom implementation of Scipy's expit
    function. If the scaling array is supplied, it does
    the unscaling keeping in mind how 0s and 1s were mapped
    to different values while the scaling was done.
    expit is the inverse of the logit function."""

    x = np.array(x_input)

    x = expit(x)

    if scaling_array is not None:

        if np.min(scaling_array) == 0:
            min_sa = np.min(scaling_array[scaling_array != 0])
            lower_lmt = min_sa / 2.0
            x[np.where(x <= lower_lmt)[0]] = 0.0

        if np.max(scaling_array) == 1:
            max_sa = np.max(scaling_array[scaling_array != 1])
            upper_lmt = 1.0 - (1.0 - max_sa) / 2.0
            x[np.where(x >= upper_lmt)[0]] = 1.0

    return x


def unscale_preds(df_input, scaling_df_path=None, drop_old=True):

    df = df_input.copy(deep=True)

    if (
        "preds_custom_logit_bt" not in df.columns.values.tolist()
        or "preds_ln_R_e_asec" not in df.columns.values.tolist()
        or "preds_ln_total_flux_adus" not in df.columns.values.tolist()
    ):
        raise ValueError(
            "This function is not appropriate for doing the transformatiuon"
        )

    if scaling_df_path is not None:
        df_scaling = pd.read_csv(scaling_df_path)
        scaling_array = np.array(df_scaling["bt"])
    else:
        scaling_array = None

    df["preds_R_e_asec"] = np.exp(df["preds_ln_R_e_asec"])
    df["preds_total_flux_adus"] = np.exp(df["preds_ln_total_flux_adus"])
    df["preds_bt"] = expit_custom(
        df["preds_custom_logit_bt"], scaling_array=scaling_array
    )
    df["preds_total_mag"] = -2.512 * np.log10(df["preds_total_flux_adus"]) + 27

    if drop_old is True:
        df = df.drop(
            columns=[
                "preds_ln_R_e_asec",
                "preds_ln_total_flux_adus",
                "preds_custom_logit_bt",
            ]
        )

    return df


def file_loader(args):

    inf_run_number, data_dir, unscale, scaling_df_path, drop_old = args

    new_df = pd.read_csv(data_dir + "inf_" + str(inf_run_number + 1) + ".csv")

    if unscale is True:
        new_df = unscale_preds(
            new_df, scaling_df_path=scaling_df_path, drop_old=drop_old
        )

    return new_df


def bayesian_inference_file_gobbler(
    data_dir, num=300, unscale=False, scaling_df_path=None, drop_old=True
):

    # read in the 1st csv file as the base data-frame
    base_frame = pd.read_csv(data_dir + "inf_1.csv")

    if unscale is True:
        base_frame = unscale_preds(
            base_frame, scaling_df_path=scaling_df_path, drop_old=drop_old
        )

    data = xr.Dataset.from_dataframe(base_frame)
    data = data.expand_dims(inf_run_number=range(num)).copy(deep=True)

    print("Loading Files.....")
    with MultiPool() as pool:
        args = list(
            zip(
                range(num),
                [data_dir] * num,
                [unscale] * num,
                [scaling_df_path] * num,
                [drop_old] * num,
            )
        )
        file_data = list(tqdm(pool.imap(file_loader, args), total=num))

    print("Creating Data Array")
    for i in tqdm(range(num)):

        new_df = file_data[i]

        new_df_img_numbers = np.array(new_df["object_id"])

        xarray_ds_img_number = np.array(
            data["object_id"][dict(inf_run_number=i)]
        )

        xarray_indexes_of_new_df = np.nonzero(
            np.isin(xarray_ds_img_number, new_df_img_numbers)
        )[0]

        for column_name in new_df.columns:
            if column_name.split("_")[0] == "preds":
                data[column_name].loc[dict(inf_run_number=i)][
                    xarray_indexes_of_new_df
                ] = np.array(new_df[column_name])

    return data


def create_summary_df(data):

    """This function takes the xarray produced by
    bayesian_inference_file_gobbler and then produces
    a dataframe with summary statistics."""

    # Creating scaffolding for final output summary df
    summary_df = data.loc[dict(inf_run_number=0)].to_dataframe()
    summary_df = summary_df.reset_index(drop=True)

    # detecting the preds columns
    preds_cols = []
    for column_name in summary_df.columns:
        if column_name.split("_")[0] == "preds":
            preds_cols.append(column_name)

    # dropping some columns not needed from the summary df
    drop_cols = preds_cols.copy()
    drop_cols.append("inf_run_number")
    summary_df = summary_df.drop(columns=drop_cols)

    x_domains = []
    y_pdfs = []

    for pred_col in preds_cols:
        preds_array = np.array(data[pred_col])
        summary_df[pred_col + "_mean"] = np.mean(preds_array, axis=0)
        summary_df[pred_col + "_median"] = np.median(preds_array, axis=0)
        summary_df[pred_col + "_std"] = np.std(preds_array, axis=0)
        summary_df[pred_col + "_skew"] = scipy.stats.skew(preds_array, axis=0)
        summary_df[pred_col + "_kurtosis"] = scipy.stats.kurtosis(
            preds_array, axis=0
        )
        print("Calculating Conf Ints for " + pred_col)
        with MultiPool() as pool:
            # mode,sig_ci,twosig_ci,threesig_ci =
            output = np.array(
                list(
                    tqdm(
                        pool.imap(calculate_ci, preds_array.T),
                        total=len(preds_array.T),
                    )
                ),
                dtype=object,
            )
        summary_df[pred_col + "_mode"] = output[:, 0]
        summary_df[pred_col + "_sig_ci"] = output[:, 1]
        summary_df[pred_col + "_twosig_ci"] = output[:, 2]
        summary_df[pred_col + "_threesig_ci"] = output[:, 3]

        x_domains.append(output[..., 4])
        y_pdfs.append(output[..., 5])

    return summary_df, np.array(x_domains), np.array(y_pdfs)


def save_pdfs(args):

    objid, out_pdfs_path, x_0, x_1, x_2, x_3, y_0, y_1, y_2, y_3 = args
    pdfs = np.stack([x_0, x_1, x_2, x_3, y_0, y_1, y_2, y_3], axis=0)

    filepath = out_pdfs_path + str(objid) + ".npy"
    np.save(filepath, pdfs)

    try:
        np.save(filepath, pdfs)
        return 0
    except Exception:
        print("Warning:Could not save PDF for ObjID:" + str(objid))
        return 1


@click.command()
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option("--num", type=int, default=500)
@click.option("--out_summary_df_path", type=click.Path(), required=True)
@click.option("--out_pdfs_path", type=click.Path(exists=True), required=True)
@click.option("--unscale/--no-unscale", default=False)
@click.option("--scaling_df_path", type=click.Path(exists=True), default=None)
@click.option("--drop_old/--no-drop_old", default=True)
def main(
    data_dir,
    num,
    out_summary_df_path,
    out_pdfs_path,
    unscale,
    scaling_df_path,
    drop_old,
):
    """A function which performs all the above steps
    necessary to prepeare the data for analysis"""

    data = bayesian_inference_file_gobbler(
        data_dir,
        num=num,
        unscale=unscale,
        scaling_df_path=scaling_df_path,
        drop_old=drop_old,
    )

    df, x_domains, y_pdfs = create_summary_df(data)
    df.to_csv(out_summary_df_path, index=False)

    objids = df["object_id"]
    out_pdfs_paths = [out_pdfs_path] * len(objids)
    x_0 = x_domains[0, :]
    x_1 = x_domains[1, :]
    x_2 = x_domains[2, :]
    x_3 = x_domains[3, :]
    y_0 = y_pdfs[0, :]
    y_1 = y_pdfs[1, :]
    y_2 = y_pdfs[2, :]
    y_3 = y_pdfs[3, :]

    args = list(
        zip(objids, out_pdfs_paths, x_0, x_1, x_2, x_3, y_0, y_1, y_2, y_3)
    )

    print("Saving PDFs as Arrays")
    with MultiPool() as pool:
        outlist = list(tqdm(pool.imap(save_pdfs, args), total=len(objids)))

    if not np.all(np.array(outlist) == 0):
        print(
            """There was an issue in saving some .npy files.
            See the list below for non-zero elemetns"""
        )
        print(outlist)


if __name__ == "__main__":

    main()
