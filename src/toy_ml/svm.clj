(ns toy-ml.svm
  (:use [clj-ml io classifiers filters])
  (:import [weka.classifiers.functions LibSVM]))

;; converted class attribute to nominal value manually...
(def iris (load-instances :arff (new java.io.File "/Users/andywu/projects/toy-ml/src/toy_ml/iris_arff.data")))
(def svm (LibSVM.))
(.setClassIndex iris 4)
(classifier-train svm iris)
(classifier-evaluate svm :dataset iris iris)
;; todo: separate dataset into training and test set
;; try different params of svm

;; (defn- separate-index [groups n]
;;   (let [ng (div groups (sum groups))
;;         before-last (map (fn [x] (clojure.contrib.math/floor (* n x)))
;;                          (drop-last ng))
;;         lst (- n (sum before-last))
;;         index-sep (concat before-last [lst])]
;;     (next (reduce (fn [x y]
;;                     (concat x [(+ (last x) y)]))
;;                   (cons [0] index-sep)))))

;; (defn separate [groups ds]
;;   (let [seps (separate-index groups (nrow ds))]
;;     (map (fn [x y]
;;            ($ (range x y) :all ds))
;;          (cons 0 (drop-last seps))
;;          seps)))
