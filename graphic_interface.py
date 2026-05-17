from image_selection import ImageSelector
from dataset_creator import DatasetCreator
import config.credentials as credentials
import labelbox as lb
import os
import tkinter as tk
import shutil

BAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rosbags")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_CAMERA_TOPICS = [
    "/camera_array/bottom/image_raw/compressed",
    "/camera_array/front/image_raw/compressed",
    "/zed/zed_node/left/image_rect_color/compressed",
    "/zed/zed_node/rgb/image_rect_color",
    "/zed/zed_node/right/image_rect_color/compressed",
    "/proc_simulation/bottom/compressed",
    "/proc_simulation/front/compressed",
]


class GraphicInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.client = lb.Client(api_key=credentials.API_KEY)
        self.project_list = [
            (project.name, project.uid) for project in self.client.get_projects()
        ]
        self.ontologies_list = [
            (ontology.name, ontology.uid)
            for ontology in self.client.get_ontologies(name_contains="")
        ]
        self.datasets_list = [
            (dataset.name, dataset.uid) for dataset in self.client.get_datasets()
        ]
        self.model_list = [
            os.path.join(MODEL_DIR, f)
            for f in os.listdir(MODEL_DIR)
            if f.endswith(".pt")
        ]
        self.bag_list = [os.path.join(BAG_DIR, dir) for dir in os.listdir(BAG_DIR)]
        self.selected_bags = self.bag_list
        self.preselection_coeff = 0.1
        self.topic_list = DEFAULT_CAMERA_TOPICS
        self.selected_project = self.project_list[0][0]
        self.selected_ontology = self.ontologies_list[0][0]
        self.selected_dataset = self.datasets_list[0][0]
        self.model_path = None
        self.define_layout()

    def make_scrollable_frame(self, parent):
        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        vscroll = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)
        scroll_frame_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def on_frame_configure(_):
            canvas.configure(scrollregion=canvas.bbox("all"))
            vscroll.grid(row=0, column=1, sticky="ns")

        def on_canvas_configure(event):
            canvas.itemconfig(scroll_frame_id, width=event.width)

        scroll_frame.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        canvas.configure(yscrollcommand=vscroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        # Do not grid vscroll initially; let on_frame_configure handle it
        return scroll_frame

    def frame_bag_layout(self):
        # Frame 1: Bag selection with checkboxes
        frame1_outer = tk.Frame(
            self, borderwidth=1, relief="ridge", width=550, height=400
        )
        frame1_outer.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        frame1 = self.make_scrollable_frame(frame1_outer)

        # Frame 1: Bag selection with checkboxes
        tk.Label(frame1, text="Select Bags:", font=("Arial", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.bag_vars = []
        for idx, bag_path in enumerate(self.bag_list):
            var = tk.BooleanVar(value=False)
            chk = tk.Checkbutton(frame1, text=os.path.basename(bag_path), variable=var)
            chk.grid(row=idx + 1, column=0, sticky="w")
            self.bag_vars.append((var, bag_path))

        for var, _ in self.bag_vars:
            var.trace_add("write", lambda *args: self.update_selected_bags())

    def frame_model_layout(self):
        # Frame 2: Model selection with radio buttons
        frame2_outer = tk.Frame(
            self, borderwidth=1, relief="ridge", width=550, height=400
        )
        frame2_outer.grid(row=0, column=2, padx=(20, 20), pady=(20, 20), sticky="nsew")
        frame2 = self.make_scrollable_frame(frame2_outer)

        tk.Label(frame2, text="Select Model:", font=("Arial", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.model_var = tk.StringVar(value="__none__")
        # Option for no model
        rb_none = tk.Radiobutton(
            frame2,
            text="No Model",
            variable=self.model_var,
            value="__none__",
            anchor="w",
        )
        self.model_var.set("__none__")  # Ensure "No Model" is selected by default
        rb_none.grid(row=1, column=0, sticky="w", padx=10)
        for idx, model_path in enumerate(self.model_list):
            rb = tk.Radiobutton(
                frame2,
                text=os.path.basename(model_path),
                variable=self.model_var,
                value=model_path,
                anchor="w",
            )
            rb.grid(row=idx + 2, column=0, sticky="w", padx=10)

        self.model_var.trace_add("write", self.update_model_var)
        # Set initial model_path
        self.model_path = None

    def frame_ontology_layout(self):
        # Frame 3: Ontology selection with radio buttons
        frame2_outer = tk.Frame(
            self, borderwidth=1, relief="ridge", width=550, height=400
        )
        frame2_outer.grid(row=1, column=2, padx=(20, 20), pady=(20, 20), sticky="nsew")
        frame2 = self.make_scrollable_frame(frame2_outer)

        # Set the first ontology as checked by default
        default_ontology = (
            self.ontologies_list[0][0] if self.ontologies_list else "__none__"
        )
        self.ontology_var = tk.StringVar(value=default_ontology)
        tk.Label(frame2, text="Select Ontology:", font=("Arial", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        for idx, (name, _) in enumerate(self.ontologies_list):
            rb = tk.Radiobutton(
                frame2, text=name, variable=self.ontology_var, value=name, anchor="w"
            )
            rb.grid(row=idx + 1, column=0, sticky="w", padx=10)

        self.ontology_var.trace_add("write", self.update_ontology_var)

    def frame_project_layout(self):
        # Frame 4: Project selection with radio buttons
        frame4_outer = tk.Frame(
            self, borderwidth=1, relief="ridge", width=550, height=400
        )
        frame4_outer.grid(row=1, column=0, padx=(20, 20), pady=(20, 20), sticky="nsew")
        frame4 = self.make_scrollable_frame(frame4_outer)

        tk.Label(frame4, text="Select Project:", font=("Arial", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.project_var = tk.StringVar(value="__new__")
        # "New Project" option with text entry
        self.new_project_var = tk.StringVar()
        rb_new = tk.Radiobutton(
            frame4,
            text="New Project:",
            variable=self.project_var,
            value="__new__",
            anchor="w",
            width=20,
        )
        rb_new.grid(row=1, column=0, sticky="w", padx=10)
        entry_new = tk.Entry(frame4, textvariable=self.new_project_var, width=15)
        entry_new.grid(row=1, column=1, sticky="w", padx=5)

        # Disable entry unless "New Project" is selected
        def project_toggle_entry(*args):
            if self.project_var.get() == "__new__":
                entry_new.config(state="normal")
            else:
                entry_new.config(state="disabled")

        self.project_var.trace_add("write", project_toggle_entry)
        entry_new.config(state="normal")
        for idx, (name, _) in enumerate(self.project_list):
            rb = tk.Radiobutton(
                frame4,
                text=name,
                variable=self.project_var,
                value=name,
                anchor="w",
                width=20,
            )
            rb.grid(row=idx + 2, column=0, sticky="w", padx=10)

        self.project_var.trace_add("write", self.update_project_var)
        self.new_project_var.trace_add("write", self.update_project_var)

    def frame_dataset_layout(self):
        # Frame 5: Dataset selection with radio buttons
        frame5_outer = tk.Frame(
            self, borderwidth=1, relief="ridge", width=550, height=400
        )
        frame5_outer.grid(row=1, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        frame5 = self.make_scrollable_frame(frame5_outer)

        tk.Label(frame5, text="Select Dataset:", font=("Arial", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        # Add "New Dataset" option with text entry
        self.dataset_var = tk.StringVar(value="__none__")
        self.new_dataset_var = tk.StringVar()
        self.dataset_var.set("__new__")
        rb_new = tk.Radiobutton(
            frame5,
            text="New Dataset:",
            variable=self.dataset_var,
            value="__new__",
            anchor="w",
            width=20,
        )
        rb_new.grid(row=1, column=0, sticky="w", padx=10)
        entry_new = tk.Entry(frame5, textvariable=self.new_dataset_var, width=15)
        entry_new.grid(row=1, column=1, sticky="w", padx=5)

        # Disable entry unless "New Dataset" is selected
        def toggle_entry(*args):
            if self.dataset_var.get() == "__new__":
                entry_new.config(state="normal")
            else:
                entry_new.config(state="disabled")

        self.dataset_var.trace_add("write", toggle_entry)
        entry_new.config(state="normal")
        # Option for no dataset
        for idx, (name, _) in enumerate(self.datasets_list):
            rb = tk.Radiobutton(
                frame5,
                text=name,
                variable=self.dataset_var,
                value=name,
                anchor="w",
                width=20,
            )
            rb.grid(row=idx + 2, column=0, sticky="w", padx=10)

        self.dataset_var.trace_add("write", self.update_dataset_var)
        self.new_dataset_var.trace_add("write", self.update_dataset_var)

    def frame_topic_layout(self):
        # Frame 6: Topic selection with checkboxes
        frame6_outer = tk.Frame(
            self, borderwidth=1, relief="ridge", width=550, height=400
        )
        frame6_outer.grid(row=0, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        frame6 = self.make_scrollable_frame(frame6_outer)

        tk.Label(frame6, text="Select Topics:", font=("Arial", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.topic_vars = []
        for idx, topic in enumerate(self.topic_list):
            var = tk.BooleanVar(value=True)
            chk = tk.Checkbutton(frame6, text=topic, variable=var)
            chk.grid(row=idx + 1, column=0, sticky="w")
            self.topic_vars.append(var)

        for var in self.topic_vars:
            var.trace_add("write", lambda *args: self.update_selected_topics())

    def frame_coeff_layout(self):
        # Frame 7: Preselection coefficient slider
        frame7_outer = tk.Frame(
            self, borderwidth=1, relief="ridge", width=550, height=100
        )
        frame7_outer.grid(row=2, column=1, padx=(20, 20), pady=(20, 20), sticky="nsew")
        frame7 = self.make_scrollable_frame(frame7_outer)

        tk.Label(
            frame7, text="Preselection Coefficient:", font=("Arial", 12, "bold")
        ).grid(row=0, column=0, sticky="w")
        self.preselection_var = tk.DoubleVar(value=self.preselection_coeff)
        slider = tk.Scale(
            frame7,
            from_=0.001,
            to=0.2,
            resolution=0.001,
            orient="horizontal",
            variable=self.preselection_var,
            length=300,
        )
        slider.grid(row=1, column=0, sticky="w", padx=10)
        self.preselection_var.trace_add("write", self.update_preselection_coeff)

    def update_model_var(self, *args):
        val = self.model_var.get()
        self.model_path = None if val == "__none__" else val

    def update_selected_model(self, *args):
        self.model_path = self.model_var.get()

    def update_selected_bags(self):
        self.selected_bags = [path for var, path in self.bag_vars if var.get()]

    def update_ontology_var(self, *args):
        val = self.ontology_var.get()
        self.selected_ontology = (
            self.ontologies_list[0][0] if val == "__none__" else val
        )

    def update_project_var(self, *args):
        val = self.project_var.get()
        if val == "__none__":
            self.selected_project = self.project_list[0][0]
        elif val == "__new__":
            self.selected_project = self.new_project_var.get()
        else:
            self.selected_project = val

    def update_dataset_var(self, *args):
        val = self.dataset_var.get()
        if val == "__none__":
            self.selected_dataset = self.datasets_list[0][0]
        elif val == "__new__":
            self.selected_dataset = self.new_dataset_var.get()
        else:
            self.selected_dataset = val

    def update_selected_topics(self):
        self.topic_list = [
            topic for var, topic in zip(self.topic_vars, self.topic_list) if var.get()
        ]
        if not self.topic_list:
            self.topic_list = DEFAULT_CAMERA_TOPICS

    def update_preselection_coeff(self, *args):
        self.preselection_coeff = self.preselection_var.get()

    def define_layout(self):
        self.geometry = (1600, 1000)
        self.title("Parameters Selection")

        self.frame_bag_layout()
        self.frame_model_layout()
        self.frame_ontology_layout()
        self.frame_project_layout()
        self.frame_dataset_layout()
        self.frame_topic_layout()
        self.frame_coeff_layout()

        launch_button = tk.Button(
            self,
            text="Launch",
            font=("Arial", 14, "bold"),
            bg="green",
            fg="white",
            command=self.launch,
        )
        launch_button.grid(row=3, column=0, columnspan=2, pady=(20, 20))
        self.mainloop()

    def launch(self):
        self.destroy()
        print(self.selected_project)
        print(self.selected_ontology)
        print(self.selected_dataset)
        print(self.model_path)
        print(self.selected_bags)

        selector = ImageSelector(
            self.selected_bags, self.preselection_coeff, self.topic_list
        )
        selector.manage_all_bags()

        print("Image selection completed. Proceeding to dataset creation...")
        dataset_creator = DatasetCreator(
            self.client,
            self.selected_project,
            self.selected_ontology,
            self.selected_dataset,
            selector.TEMP_DIR,
            self.model_path,
        )
        dataset_creator.create_dataset()

        shutil.rmtree(selector.TEMP_DIR)

    