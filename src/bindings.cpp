#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "knn.hpp"


namespace py = pybind11;

PYBIND11_MODULE(_classifier, m) {
    
    m.doc() = "Classification library";

    py::class_<classifier_std::KNNClassifier>(m, "KNNClassifier")
        .def(py::init<int>(), py::arg("k") = 5)

        .def("fit",
            &classifier_std::KNNClassifier::fit,
            py::arg("X"),
            py::arg("y"),
            "Fit"    
        )

        .def("predict",
            &classifier_std::KNNClassifier::predict,
            py::arg("X"),
            "Predict labels"
        )

        .def("is_fitted",
            &classifier_std::KNNClassifier::is_fitted,
            "Is model fitted"
        );
}