(ns toy-ml.applications.car
  (:use [toy-ml dtree bagging]
        [incanter core io]))

(def car-ds (read-dataset "/Users/andywu/projects/toy-ml/src/toy_ml/applications/car.data" :header true))

(def car-train ($ (range 0 (nrow car-ds) 2) :all car-ds))

(def car-test ($ (range 1 (nrow car-ds) 2) :all car-ds))

(defn dtree-accuracy [car-train car-test]
  (let [out (dtree-classify-dataset (dtree car-train :class)
                                    car-test)
        true-val ($ :class car-test)
        result (map #(if (=%1 %2) 1 0)
                    out true-val)]
    (/ (sum result) (count result))))

(defn bagging-accuracy [car-train car-test n-samples]
  (let [samples (repeatedly n-samples #(sample car-train))
        out (bagging-classify car-test
                              (bagging-classifiers samples dtree :class))
        true-val ($ :class car-test)
        result (map #(if (=%1 %2) 1 0)
                    out true-val)]
    (/ (sum result) (count result))))
