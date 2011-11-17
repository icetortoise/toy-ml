(ns toy-ml.applications.iris
  (:require clojure.contrib.math))
(use '(toy-ml core mlp))
(use '(incanter core io stats))


;; load, randomize, train and test with confmat
(def path-to-data
  "/Users/andywu/projects/toy-ml/src/toy_ml/applications/iris_proc.data")

(def ds (read-dataset path-to-data))

;; todo: think about macros to help?
(defn run-iris [ds]
  (let [shuffled (to-matrix (randomize ds))
        origin-inputs ($ :all (range (dec (ncol shuffled)))
                         shuffled)
        origin-targets (one-to-n ($ :all (dec (ncol shuffled))
                                    shuffled))
        [train-inputs test-inputs]
        (separate [100 50] origin-inputs)
        [train-targets test-targets]
        (separate [100 50] origin-targets)
        weights (mlp-train train-inputs train-targets [3] 0.01 0.02
                           (make-logistic 1) (end-after-iter 1500))]
    (correct-percentage 
     (last (mlp-forward test-inputs weights (:forward (make-logistic 1))))
     test-targets)))
        
