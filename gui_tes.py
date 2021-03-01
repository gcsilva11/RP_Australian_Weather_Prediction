from tkinter import *
from main import run

"""
 HELP SECTION - Explanation of different functions and their parameters

    clicked() - function called when GUI button is pressed to start the main script
    
    run(region, sel, sel_n, red, red_n, clas) - runs with the passed GUI parameters
        region - region to select from the data set
        sel - feature selector
        sel_n - feature number after selection
        red - feature reductor
        red_n - number of features to be reduced
        clas - classifier
        
    data_preprocessing(dataset_name, dummies, region, plots) - pre processing of the panda dataframe.
        dataset_name - the name of the dataset to be processed
        dummies - determine if dummies should be used or factorization instead
        region - the region that is to be chosen for modeling
        plots - plot a heatmap of the dataset
    
    feature_redux_and_classify(df, target, selection, reduction, classifier, n_features, n_reduction) - reduction of features and classification of model
        df - dataframe to be classified
        selection - selecton method
        reduction - reduction method
        classifier - classifier chosen
        n_features - number of features to remain
        n_reduction - number of features to reduce


"""
# Function called when the GUI run button is clicked.
def clicked():
    run(region.get(), feat_sel.get(), feat_sel_num.get(), feat_red.get(), feat_red_num.get(), classifier.get())


# GUI creation and running
if __name__ == '__main__':

    window = Tk()
    window.title("Rp - Weather Prediction")

    lbl = Label(window, text="Australian State:")
    lbl.grid(column=0, row=0)
    region = StringVar(window)
    region.set("All")
    w = OptionMenu(window, region, "All", "New South Wales", "Victoria", "Queensland", "Western Australia", "South Australia", "Tasmania", "Northern Territory")
    w.grid(column=2, row=0)

    lbl = Label(window, text="Feature Selection:")
    lbl.grid(column=0, row=2)
    feat_sel = StringVar(window)
    feat_sel.set("None")
    w = OptionMenu(window, feat_sel, "None", "Kruskal-Wallis", "ROC", "K-Best", "RFE")
    w.grid(column=2, row=2)

    feat_sel_num = Entry(window, width=10)
    feat_sel_num.grid(column=3, row=2)

    lbl = Label(window, text="Feature Reduction:")
    lbl.grid(column=0, row=3)
    feat_red = StringVar(window)
    feat_red.set("None")
    w = OptionMenu(window, feat_red, "None", "PCA", "LDA")
    w.grid(column=2, row=3)

    feat_red_num = Entry(window, width=10)
    feat_red_num.grid(column=3, row=3)

    lbl = Label(window, text="Classifier:")
    lbl.grid(column=0, row=4)
    classifier = StringVar(window)
    classifier.set("Euclidean")
    w = OptionMenu(window, classifier, "Euclidean", "Mahalonobis", "LDA", "Bayes", "K-Nearest", "SVC", "Parzen Window")
    w.grid(column=2, row=4)

    btn = Button(window, text="RUN", command= clicked)
    btn.grid(column=3, row=7)
    window.mainloop()