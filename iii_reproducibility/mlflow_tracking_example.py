import numpy
import sklearn.naive_bayes
import sklearn.feature_extraction.text
import sklearn.pipeline

# New additions
import mlflow.sklearn
mlflow.set_tracking_uri("http://atrium.datmo.com")
mlflow.set_experiment("training_module")

...

def train_and_evaluate_model():
    with mlflow.start_run():
        # Load dataset:
        docs, labels = load_labeled_data_set()
        train_docs, train_labels, test_docs, test_labels = partition_data_set(docs, labels)

        # Train classifier:
        mlflow.log_param('classifier', "naive bayes")
        mlflow.log_param('code commit id', version)
        classifier = sklearn.pipeline.Pipeline([
            ("vect", sklearn.feature_extraction.text.CountVectorizer()),
            ("tfidf", sklearn.feature_extraction.text.TfidfTransformer()),
            ("clf", sklearn.naive_bayes.MultinomialNB()),
        ])
        classifier.fit(train_docs, train_labels)

        # Evaluate classifier:
        predicted_labels = classifier.predict(test_docs)
        accuracy = numpy.mean(predicted_labels == test_labels)
        print("Accuracy = %s" % (accuracy,))
        mlflow.log_metric('accuracy', accuracy)

        mlflow.sklearn.log_model(classifier, "model")

train_and_evaluate_model()