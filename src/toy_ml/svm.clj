(ns toy-ml.svm
  (:use [clj-ml io classifiers filters]
        [toy-ml core])
  (:import [weka.classifiers.functions LibSVM]
           [weka.core SelectedTag]))

;; converted class attribute to nominal value manually...
(def iris (load-instances :arff (new java.io.File "/Users/andywu/projects/toy-ml/src/toy_ml/iris_arff.data")))

;; implement clj-ml wrapper for LibSVM
(defmethod make-classifier [:libsvm :libsvm]
  ([kind algo & options]
     ))

(defn run-svm-iris [ds]
  (let [svm (LibSVM.)]
    (.setClassIndex iris 4)
    (classifier-train svm iris)
    (classifier-evaluate svm :dataset iris iris)
    ;; using a polynomial kernel with degree 0/1/2/3/10
    (try-params :params [d [0 1 2 3 10]
                         type [(SelectedTag. LibSVM/KERNELTYPE_POLYNOMIAL LibSVM/TAGS_KERNELTYPE )]]
                :temp-bindings [temp (do (.setKernelType svm  type)
                                         (.setDegree svm d)
                                         (classifier-train svm iris))
                                evaluated  (classifier-evaluate svm :dataset iris iris)]
                :score  (:correct evaluated)
                :return [d])))

