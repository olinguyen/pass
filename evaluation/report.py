from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pickle

class ClassificationReporting:
    metrics = [accuracy_score, f1_score, confusion_matrix, classification_report]

    def __init__(self, clf, X, y, cv=False):
        """
        :param clf:
        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :param n_classes:
        :return:
        """
        self.clf = clf
        self.X = X
        self.y = y
        self.cv = cv
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        """

        self.report = {}

    def serialize_classifier(self):
        #self.report["type"] = type(self.clf.named_steps["clf"]).__name__
        self.report["clf"] = pickle.dumps(self.clf)

    def set_name(self, name):
        """
        :param name:
        :return:
        """
        self.report["name"] = name
        return self

    def set_level(self, level):
        """
        :param level:
        :return:
        """
        self.report["level"] = level
        return self

    def set_notes(self, notes):
        """
        :param notes:
        :return:
        """
        self.report["notes"] = notes
        return self

    def set_params(self, attr_name, attr):
        """
        :param attr_name:
        :param attr:
        :return:
        """
        self.report[attr_name] = attr
        return self

    def set_datetime(self):
        """
        :return:
        """
        self.report["created_at"] = str(datetime.today())
        return self

    def compute_metrics(self, prefix, X, y):
        """
        :param prefix:
        :param X:
        :param y:
        :return:
        """

        if self.cv:
            kf = StratifiedKFold(n_splits=5, random_state=42)
            for i, train, test in enumerate(kf.split(self.X, self.y)):
                X_train, y_train = X[train], y[train]
                X_test, y_test = X[test], y[test]
                pass
        else:
            y_predict = self.clf.predict(X)
            results = {}
            # simple metrics
            for metric in self.metrics:
                kwargs = {}
                if metric == f1_score:
                    kwargs["average"] = "weighted"
                result = metric(y_true=y, y_pred=y_predict, **kwargs)
                results[metric.__name__] = result.tolist() if hasattr(result, "tolist") else result
            self.report[prefix] = results

    def set_path(self):
        name_template = "accuracy:{accuracy}|f1:{f1_score}"
        self.report["path"] = name_template.format(
            #level=self.report["level"],
            accuracy=self.report["testing_results"]["accuracy_score"],
            f1_score=self.report["testing_results"]["f1_score"],
        )
        return self

    def compute_rocauc(self,):
        """
        :return:
        """
        # Binarize the output
        # y_test = label_binarize(self.y_test, classes=list(range(self.n_classes)))
        # Compute ROC curve and ROC area for each class
        if self.cv:
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            folds = []

            kf = StratifiedKFold(n_splits=5, random_state=42)
            for i, train, test in enumerate(kf.split(X, y)):
                probas_ = classifier.fit(self.X[train], self.y[train]).predict_proba(self.X[test])
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                folds.append(dict(fpr=fpr,
                                  tpr=tpr,
                                  roc_auc=roc_auc))

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            std_tpr = np.std(tprs, axis=0)


            self.report["roc_auc"] = dict(
                #fpr={str(k): v.tolist() for k, v in fpr.items()},
                mean_tpr=mean_tpr,
                mean_auc=mean_auc,
                std_auc=std_auc,
                std_tpr=std_tpr,
                folds=folds
            )
        else:
            y_score = self.clf.predict_proba(self.X_test)

            fpr, tpr, _ = roc_curve(self.y_test, y_score[:, 1])
            roc_auc = auc(fpr, tpr)

            self.report["roc_auc"] = dict(
                #fpr={str(k): v.tolist() for k, v in fpr.items()},
                fpr=fpr,
                tpr=tpr,
                roc_auc=roc_auc
            )

    def print(self):
        """
        :return:
        """
        print("Training Results")
        print("~~~~~~~~~~~~~~~~")
        for k, v in self.report["training_results"].items():
            print(k, v, "\n", sep="\n")
        print()
        print()
        print("Testing Results Results")
        print("~~~~~~~~~~~~~~~~~~~~~~~")
        for k, v in self.report["testing_results"].items():
            print(k, v, "\n", sep="\n")
        print()
        print()

    def show(self):
        """
        :return:
        """
        fpr = self.report["roc_auc"]["fpr"]
        tpr = self.report["roc_auc"]["tpr"]
        roc_auc = self.report["roc_auc"]["roc_auc"]

        plt.figure()

        plt.plot(fpr, tpr, label='ROC curve of class {0} (area = {1:0.2f})'.format('0', roc_auc))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for {}'.format(self.report.get("name", "Classifier")))
        plt.legend(loc="lower right")
        plt.show()

    def create_report(self, output=False, show_roc=False):
        """
        :param output:
        :param show_roc:
        :return:
        """

        if not self.cv:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
            self.X_test = X_test
            self.y_test = y_test
            self.compute_metrics("training_results", X_train, y_train)
            self.compute_metrics("testing_results", X_test, y_test)
        else:
            pass

        try:
            self.compute_rocauc()
        except:
            pass

        self.serialize_classifier()
        self.set_path()

        if output:
            self.print()

        if show_roc:
            self.show()

        return self.report

X_train, X_test, y_train, y_test = train_test_split(X_feats, y_sb)
lr.fit(X_train, y_train)
report = ClassificationReporting(lr, X_feats, y_sb, cv=False)
