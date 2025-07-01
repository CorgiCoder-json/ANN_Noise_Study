from heatmap_analysis import Report, gather_model_paths, load_references, open_log

if __name__ == "__main__":
    class_path = "D:\\model_dataset\\classification"
    regress_path = "D:\\model_dataset\\regression"
    model_paths = gather_model_paths(class_path, regress_path)
    class_models = load_references(model_paths[1])
    regress_models = load_references(model_paths[0])