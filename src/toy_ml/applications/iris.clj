(ns toy-ml.applications.iris
  (:require clojure.contrib.math))
(use '(toy-ml core mlp))
(use '(incanter core io stats))


;; load, randomize, train and test with confmat
(def path-to-data
  "/Users/andywu/projects/toy-ml/src/toy_ml/applications/iris_proc.data")

(def ds (read-dataset path-to-data))

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
        [weights costs] (mlp-train train-inputs train-targets [3] 0.01 0.02
                                   (make-logistic 1) (end-after-iter 1000))]
    (correct-percentage 
     (mlp-recall test-inputs weights (:forward (make-logistic 1)))
     test-targets)))

(comment ;sample usage of try-params
(def evaluated   (let [shuffled (to-matrix (randomize ds))
                       origin-inputs ($ :all (range (dec (ncol shuffled)))
                                        shuffled)
                       origin-targets (one-to-n ($ :all (dec (ncol shuffled))
                                                   shuffled))
                       [inputs inputs-val]
                       (separate [100 50] origin-inputs)
                       [targets targets-val]
                       (separate [100 50] origin-targets)]
                   (try-params {:params 
                                [hidden [[3] [4]]
                                 reg-coff [0.01 0.03]
                                 l-rate [0.02 0.06]]
                                :temp-bindings
                                [[weights costs] (mlp-train inputs targets hidden reg-coff l-rate
                                                    (make-logistic 1) (end-after-iter 1000))
                                 outputs-val (mlp-recall inputs-val weights
                                                         (:forward (make-logistic 1)))
                                 recall (fn [inputs] (mlp-recall inputs weights
                                                                 (:forward (make-logistic 1))))
                                 ]
                                :score (correct-percentage outputs-val targets-val)
                                :return [hidden reg-coff l-rate recall]})))
)