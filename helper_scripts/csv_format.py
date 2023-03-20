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
            if row[0].count(":") > 1:
                sorted_data = sort_data(row)
            else:
                sorted_data = row
            if i % 5 == 0:
                accumulation.append(sorted_data)
            elif i % 5 == 1:
                example.append(sorted_data)
            elif i % 5 == 2:
                margin.append(sorted_data)
            elif i % 5 == 3:
                clause.append(sorted_data)
            elif i % 5 == 4:
                specificity.append(sorted_data)
    with open("results_ordered.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow("")

        write_section(example)
        write_section(margin)
        write_section(clause)
        write_section(specificity)
        write_section(accumulation)


def write_section(section):
    with open("results_ordered.csv", mode='a') as f:
        writer = csv.writer(f)
        for item in section:
            writer.writerow(item)


def sort_data(row):
    data = row[0].split("\n")
    data = data[1::]

    return data


if __name__ == "__main__":
    format_csv("results.csv")
