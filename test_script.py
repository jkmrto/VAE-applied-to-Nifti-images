for evaluation in evaluation_type:
    for execution in execution_type:
        for kernel in kernel_size:
            selectec_rows = []
            for row in list_rows:
                if row[" evaluation"] == evaluation:
                    if row[" test|train"] == execution:
                        if row[" fold"] == str(fold_selected):
                            plt.figure(int(row["kernel_size"]))
                            if row["kernel_size"] == str(kernel):
                                print("selected")
                                fpr = \
                                    [float(str_number) for str_number in
                                     row[" false_positive_rate"].split(",")]
                                print(fpr)
                                tpr = \
                                    [float(str_number) for str_number in
                                     row[" true_positive_rate"].split(",")]
                                plt.plot(np.array(fpr), np.array(tpr))

                            img_idi = "{0},{1},{2}".format(
                                    row[" evaluation"],
                                    row[" test|train"],
                                    row[" fold"])
                            plt.title(img_idi)

                            plt.savefig(os.path.join(path_to_folder,
                                        "{}.png".format(img_idi)))