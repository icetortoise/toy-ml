(ns toy-ml.svm
  (:use [clj-ml io classifiers filters options-utils]
        [toy-ml core])
  (:import [weka.classifiers.functions LibSVM]
           [weka.core SelectedTag]))

;; converted class attribute to nominal value manually...
(def iris (load-instances :arff (new java.io.File "/Users/andywu/projects/toy-ml/src/toy_ml/iris_arff.data")))

;; clj-ml compatible LibSVM classifier
(defmethod make-classifier-options [:LibSVM :LibSVM]
  ([kind _ options]
     ;; these options are copied from LibSVM website, looks like it is not consistent with what are actually
     ;; by the java wrapper. Fix this..
     (let [cols-val-a (check-option-values {:svm-type "-S"
                                            :kernel-type "-K"
                                            :degree "-D"
                                            :gamma "-G"
                                            :coef-zero "-R"
                                            :cost "-C"
                                            :nu "-N"
                                            :loss-epsilon "-P"
                                            :cache-size "-M"
                                            :tolerance-epsilon "-E"
                                            :shrinking "-H"
                                            :prob-esti "-B"
                                            :weight "-W"}
                                           options
                                           [])]
       (into-array (reverse (map str cols-val-a))))))

(defmethod make-classifier [:LibSVM :LibSVM]
  ([kind _ & options]
     (let [options-read (if (empty? options) {} (first options))
           classifier (new LibSVM)
           opts (make-classifier-options :LibSVM :LibSVM options-read)]
       (.setOptions classifier opts)
       classifier)))
       
(defn run-svm-iris [ds]
  (.setClassIndex iris 4)
  ;; using a polynomial kernel with degree 0/1/2/3/10
  (try-params :params [degree [0 1 2 3 10]
                       kernel-type [1 2]]
              :temp-bindings [svm (make-classifier :LibSVM :LibSVM
                                                   {:kernel-type kernel-type
                                                    :degree degree})
                              temp (classifier-train svm iris)
                              evaluated  (classifier-evaluate svm :dataset iris iris)]
              :score  (:correct evaluated)
              :return [degree kernel-type]))

