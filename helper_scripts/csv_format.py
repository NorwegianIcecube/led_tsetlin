import csv


def format_csv(file_name):

    with open(file_name, 'r') as f:
        example = ["examples changed"]
        margin = ["margin changed"]
        clause = ["clause changed"]
        specificity = ["specificity changed"]
        accumulation = ["accumulation changed"]
        reader = csv.reader(f)
        i = 0
        for j, row in enumerate(reader):
            if j % 6 == 0:
                i += 1

            if i % 5 == 0:
                accumulation.append(row)
            elif i % 5 == 1:
                example.append(row)
            elif i % 5 == 2:
                margin.append(row)
            elif i % 5 == 3:
                clause.append(row)
            elif i % 5 == 4:
                specificity.append(row)

    with open("results_ordered.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(example)
        writer.writerow(margin)
        writer.writerow(clause)
        writer.writerow(specificity)
        writer.writerow(accumulation)


if __name__ == "__main__":
    format_csv("results.csv")
