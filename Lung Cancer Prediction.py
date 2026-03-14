# ============================================================
#  Lung Cancer Prediction using CNN + Transfer Learning
#  Frontend: Tkinter GUI — single-file application
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

# ── Dataset / model paths ───────────────────────────────────
train_folder    = 'dataset/train'
test_folder     = 'dataset/test'
validate_folder = 'dataset/valid'

normal_folder               = '/normal'
adenocarcinoma_folder       = '/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib'
large_cell_carcinoma_folder = '/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa'
squamous_cell_carcinoma_folder = '/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'

MODEL_PATH = 'trained_lung_cancer_model.h5'
IMAGE_SIZE = (350, 350)
OUTPUT_SIZE = 4

CLASS_INFO = {
    'normal': {
        'label': 'Normal',
        'detail': 'No malignancy detected',
        'color': '#2d7a2d',
        'bg': '#eaf5ea',
    },
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': {
        'label': 'Adenocarcinoma',
        'detail': 'Left lower lobe · T2 N0 M0 · Stage Ib',
        'color': '#b03030',
        'bg': '#fceaea',
    },
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': {
        'label': 'Large Cell Carcinoma',
        'detail': 'Left hilum · T2 N2 M0 · Stage IIIa',
        'color': '#b06010',
        'bg': '#fef3e2',
    },
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': {
        'label': 'Squamous Cell Carcinoma',
        'detail': 'Left hilum · T1 N2 M0 · Stage IIIa',
        'color': '#7030a0',
        'bg': '#f5eafa',
    },
}


# ── Lazy-load TensorFlow so the GUI opens instantly ─────────
_model = None
_class_labels = None

def get_model():
    global _model, _class_labels
    if _model is not None:
        return _model, _class_labels

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

    # ── Try loading a saved model first ────────────────────
    if os.path.exists(MODEL_PATH):
        _model = load_model(MODEL_PATH)
        _class_labels = list(CLASS_INFO.keys())
        return _model, _class_labels

    # ── Otherwise build + (optionally) train ───────────────
    pretrained_model = tf.keras.applications.Xception(
        weights='imagenet', include_top=False,
        input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = False

    _model = Sequential([
        pretrained_model,
        GlobalAveragePooling2D(),
        Dense(OUTPUT_SIZE, activation='softmax'),
    ])
    _model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    # ── Train if dataset exists ─────────────────────────────
    if os.path.isdir(train_folder) and os.path.isdir(test_folder):
        train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
        test_datagen  = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_folder, target_size=IMAGE_SIZE,
            batch_size=8, color_mode='rgb', class_mode='categorical',
        )
        validation_generator = test_datagen.flow_from_directory(
            test_folder, target_size=IMAGE_SIZE,
            batch_size=8, color_mode='rgb', class_mode='categorical',
        )
        _class_labels = list(train_generator.class_indices.keys())

        callbacks = [
            ReduceLROnPlateau(monitor='loss', patience=5, verbose=0,
                              factor=0.5, min_lr=1e-6),
            EarlyStopping(monitor='loss', min_delta=0,
                          patience=6, verbose=0, mode='auto'),
            ModelCheckpoint(filepath='best_model.weights.h5',
                            verbose=0, save_best_only=True,
                            save_weights_only=True),
        ]
        _model.fit(
            train_generator,
            steps_per_epoch=25,
            epochs=50,
            callbacks=callbacks,
            validation_data=validation_generator,
            validation_steps=20,
            verbose=1,
        )
        _model.save(MODEL_PATH)
    else:
        _class_labels = list(CLASS_INFO.keys())

    return _model, _class_labels


def load_and_preprocess_image(img_path):
    from tensorflow.keras.preprocessing import image as kimage
    img = kimage.load_img(img_path, target_size=IMAGE_SIZE)
    arr = kimage.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr


def predict_image(img_path):
    model, class_labels = get_model()
    arr         = load_and_preprocess_image(img_path)
    predictions = model.predict(arr, verbose=0)[0]
    idx         = int(np.argmax(predictions))
    label       = class_labels[idx]
    return label, predictions, class_labels


# ═══════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════

DARK   = '#1a1a1a'
LIGHT  = '#f7f5f0'
BORDER = '#dedad2'
ACCENT = '#1a1a1a'
MUTED  = '#888888'
WHITE  = '#ffffff'
FONT_H = ('Helvetica', 13, 'bold')
FONT_B = ('Helvetica', 11)
FONT_S = ('Courier', 9)


class LungCancerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Lung Cancer Classifier · Xception')
        self.configure(bg=LIGHT)
        self.resizable(True, True)
        self.minsize(700, 540)

        self._img_path  = None
        self._photo_ref = None   # keep reference to avoid GC

        self._build_ui()

    # ── UI construction ────────────────────────────────────
    def _build_ui(self):
        # ── Header ─────────────────────────────────────────
        hdr = tk.Frame(self, bg=DARK, padx=24, pady=14)
        hdr.pack(fill='x')

        tk.Label(hdr, text='Lung Cancer Classifier',
                 font=('Helvetica', 15, 'bold'),
                 bg=DARK, fg=WHITE).pack(side='left')

        tk.Label(hdr, text='CNN + Transfer Learning · Xception · 4 classes',
                 font=('Courier', 9), bg=DARK, fg='#888888').pack(
                     side='left', padx=(12, 0), pady=(4, 0))

        # ── Main layout ────────────────────────────────────
        body = tk.Frame(self, bg=LIGHT)
        body.pack(fill='both', expand=True, padx=24, pady=20)

        left  = tk.Frame(body, bg=LIGHT)
        right = tk.Frame(body, bg=LIGHT)
        left.pack(side='left', fill='both', expand=True, padx=(0, 16))
        right.pack(side='right', fill='both', expand=True)

        # ── Left: image panel ──────────────────────────────
        self._build_image_panel(left)

        # ── Right: result panel ────────────────────────────
        self._build_result_panel(right)

        # ── Status bar ────────────────────────────────────
        self.status_var = tk.StringVar(value='Ready — open a CT scan image to begin')
        status = tk.Label(self, textvariable=self.status_var,
                          font=('Courier', 9), bg=BORDER, fg=MUTED,
                          anchor='w', padx=14, pady=5)
        status.pack(fill='x', side='bottom')

    def _build_image_panel(self, parent):
        tk.Label(parent, text='CT SCAN IMAGE',
                 font=('Courier', 9), bg=LIGHT, fg=MUTED).pack(anchor='w')

        # Drop zone / preview area
        self.img_frame = tk.Frame(parent, bg=WHITE,
                                  highlightbackground=BORDER,
                                  highlightthickness=1,
                                  width=300, height=300)
        self.img_frame.pack(fill='both', expand=True, pady=(6, 0))
        self.img_frame.pack_propagate(False)

        self.placeholder_lbl = tk.Label(
            self.img_frame,
            text='Click "Open Image" below\nto load a CT scan',
            font=('Helvetica', 11), bg=WHITE, fg='#cccccc',
            justify='center',
        )
        self.placeholder_lbl.place(relx=0.5, rely=0.5, anchor='center')

        self.img_label = tk.Label(self.img_frame, bg=WHITE)
        self.img_label.place(relx=0.5, rely=0.5, anchor='center')

        # Buttons row
        btn_row = tk.Frame(parent, bg=LIGHT)
        btn_row.pack(fill='x', pady=(10, 0))

        tk.Button(btn_row, text='Open Image',
                  font=FONT_B, bg=DARK, fg=WHITE,
                  activebackground='#333', activeforeground=WHITE,
                  relief='flat', padx=14, pady=7, cursor='hand2',
                  command=self._open_image).pack(side='left', padx=(0, 8))

        self.predict_btn = tk.Button(
            btn_row, text='Run Prediction',
            font=FONT_B, bg='#2b6cc8', fg=WHITE,
            activebackground='#1a50a0', activeforeground=WHITE,
            relief='flat', padx=14, pady=7, cursor='hand2',
            state='disabled', command=self._run_prediction,
        )
        self.predict_btn.pack(side='left')

        # File info
        self.file_info_var = tk.StringVar(value='No file selected')
        tk.Label(parent, textvariable=self.file_info_var,
                 font=('Courier', 9), bg=LIGHT, fg=MUTED).pack(
                     anchor='w', pady=(6, 0))

    def _build_result_panel(self, parent):
        tk.Label(parent, text='PREDICTION RESULT',
                 font=('Courier', 9), bg=LIGHT, fg=MUTED).pack(anchor='w')

        # Result card
        self.result_card = tk.Frame(parent, bg=WHITE,
                                    highlightbackground=BORDER,
                                    highlightthickness=1)
        self.result_card.pack(fill='x', pady=(6, 0))

        self.result_class_var  = tk.StringVar(value='—')
        self.result_detail_var = tk.StringVar(value='Run a prediction to see results')

        tk.Label(self.result_card, textvariable=self.result_class_var,
                 font=('Helvetica', 16, 'bold'), bg=WHITE, fg=DARK,
                 anchor='w', padx=16).pack(fill='x', pady=(14, 2))

        self.result_detail_lbl = tk.Label(
            self.result_card, textvariable=self.result_detail_var,
            font=('Courier', 9), bg=WHITE, fg=MUTED, anchor='w',
            padx=16,
        )
        self.result_detail_lbl.pack(fill='x', pady=(0, 14))

        # Confidence bars
        tk.Label(parent, text='CLASS PROBABILITIES',
                 font=('Courier', 9), bg=LIGHT, fg=MUTED).pack(
                     anchor='w', pady=(18, 6))

        self.bars_frame = tk.Frame(parent, bg=LIGHT)
        self.bars_frame.pack(fill='x')
        self._init_bars()

        # Model info
        info_frame = tk.Frame(parent, bg='#f0ece4',
                              highlightbackground=BORDER,
                              highlightthickness=1)
        info_frame.pack(fill='x', pady=(20, 0))

        notes = [
            'Model: Xception pretrained on ImageNet',
            'Input: 350×350 RGB, normalised to [0,1]',
            'Output: Softmax over 4 classes (argmax)',
            'For research use only — not clinical advice',
        ]
        for n in notes:
            tk.Label(info_frame, text=f'· {n}',
                     font=('Courier', 9), bg='#f0ece4', fg=MUTED,
                     anchor='w', padx=12, pady=2).pack(fill='x')
        tk.Label(info_frame, text='', bg='#f0ece4').pack()

    def _init_bars(self):
        self._bar_widgets = {}
        for key, info in CLASS_INFO.items():
            row = tk.Frame(self.bars_frame, bg=LIGHT)
            row.pack(fill='x', pady=3)

            tk.Label(row, text=info['label'],
                     font=('Helvetica', 10), bg=LIGHT, fg=DARK,
                     width=22, anchor='w').pack(side='left')

            track = tk.Frame(row, bg=BORDER, height=6)
            track.pack(side='left', fill='x', expand=True, padx=(4, 8))
            track.pack_propagate(False)

            fill = tk.Frame(track, bg='#cccccc', height=6)
            fill.place(x=0, y=0, relheight=1, relwidth=0)

            pct_var = tk.StringVar(value='—')
            tk.Label(row, textvariable=pct_var,
                     font=('Courier', 9), bg=LIGHT, fg=MUTED,
                     width=5, anchor='e').pack(side='left')

            self._bar_widgets[key] = (track, fill, pct_var)

    # ── Actions ────────────────────────────────────────────
    def _open_image(self):
        path = filedialog.askopenfilename(
            title='Select CT scan image',
            filetypes=[('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff'),
                       ('All files', '*.*')],
        )
        if not path:
            return
        self._img_path = path
        self._display_image(path)
        fname = os.path.basename(path)
        size  = os.path.getsize(path)
        self.file_info_var.set(f'{fname}  ·  {size/1024:.1f} KB')
        self.predict_btn.config(state='normal')
        self.status_var.set(f'Image loaded: {fname}')

        # Reset results
        self.result_class_var.set('—')
        self.result_detail_var.set('Press "Run Prediction" to classify')
        self.result_detail_lbl.config(bg=WHITE, fg=MUTED)
        self.result_card.config(bg=WHITE)
        for key, (track, fill, pct_var) in self._bar_widgets.items():
            fill.place(relwidth=0)
            fill.config(bg='#cccccc')
            pct_var.set('—')

    def _display_image(self, path):
        img = Image.open(path)
        # Fit inside the frame
        fw = self.img_frame.winfo_width()  or 300
        fh = self.img_frame.winfo_height() or 300
        img.thumbnail((fw - 4, fh - 4), Image.LANCZOS)
        self._photo_ref = ImageTk.PhotoImage(img)
        self.placeholder_lbl.place_forget()
        self.img_label.config(image=self._photo_ref)

    def _run_prediction(self):
        if not self._img_path:
            return
        self.predict_btn.config(state='disabled', text='Running…')
        self.status_var.set('Running prediction…')
        self.update_idletasks()
        self.after(50, self._do_predict)

    def _do_predict(self):
        try:
            label, probs, class_labels = predict_image(self._img_path)
        except Exception as e:
            self.status_var.set(f'Error: {e}')
            self.predict_btn.config(state='normal', text='Run Prediction')
            return

        # ── Update result card ──────────────────────────────
        info = CLASS_INFO.get(label, {
            'label': label, 'detail': '', 'color': DARK, 'bg': WHITE,
        })
        self.result_class_var.set(info['label'])
        self.result_detail_var.set(info['detail'] or label)
        self.result_card.config(bg=info['bg'])
        for w in self.result_card.winfo_children():
            w.config(bg=info['bg'], fg=info['color'])

        # ── Update bars ────────────────────────────────────
        for i, key in enumerate(CLASS_INFO.keys()):
            track, fill, pct_var = self._bar_widgets[key]
            p = float(probs[i]) if i < len(probs) else 0.0
            fill.config(bg=info['color'] if key == label else '#cccccc')
            fill.place(relwidth=p)
            pct_var.set(f'{p*100:.1f}%')

        self.status_var.set(
            f'Predicted: {info["label"]}  '
            f'({float(probs[np.argmax(probs)])*100:.1f}% confidence)'
        )
        self.predict_btn.config(state='normal', text='Run Prediction')


# ── Entry point ─────────────────────────────────────────────
if __name__ == '__main__':
    app = LungCancerApp()
    app.mainloop()