#! /usr/bin/env python
# -- coding: utf-8 --
import sys, os, warnings
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from srrf_cupy.ui.ui_srrf_form import Ui_MainWindow
from srrf_cupy.srrf import read_images_from_folder, read_images_from_tiff_file, srrf
import numpy as np
from PIL import Image


class ProgressBarThread(QThread):
    progress_update = pyqtSignal(int)
    error_signal = pyqtSignal(Exception)
    warning_signal = pyqtSignal(list)
    result = pyqtSignal(object)

    def __init__(self, func, arg):
        QThread.__init__(self)
        self.func = func
        self.arg = arg

    def __del__(self):
        self.wait()

    def run(self):
        warnings.filterwarnings('ignore', category=FutureWarning)
        with warnings.catch_warnings(record=True) as w:
            try:
                if isinstance(self.arg, dict):
                    self.result.emit(self.func(**self.arg, signal=self.progress_update))
                else:
                    self.result.emit(self.func(self.arg, signal=self.progress_update))
            except ValueError as e:
                self.error_signal.emit(e)
            else:
                self.progress_update.emit(0)
        if w:  # has warning
            self.warning_signal.emit(w)


class MainForm(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)
        # 记录默认值
        self.default_values = (self.mag_spin_box.value(),
                               self.rad_spin_box.value(),
                               self.axes_spin_box.value(),
                               self.n_frame_spin_box.value(),
                               self.dc_checkbox.checkState(),
                               self.iw_checkbox.checkState(),
                               self.gw_checkbox.checkState(),
                               self.tra_radio_button.isChecked(),
                               self.trac_radio_button.isChecked(),
                               self.trac_order_spin_box.value(),
                               self.trac_delay_spin_box.value()
                               )
        self.image_mat = None
        self.result_mat = None
        self.progress_thread = None
        self.busy = False
        # 由于部分功能有问题，隐藏图像显示区域的按钮和右键菜单
        for each in (self.pyqtimv, self.pyqtimv_2):
            each.ui.menuBtn.setVisible(False)
            each.ui.roiBtn.setVisible(False)
            each.getView().setMenuEnabled(False)
        # 把结果显示的直方图条弄得宽一点，免得gamma控件被缩放得太小
        self.pyqtimv_2.getHistogramWidget().setMinimumWidth(140)

        # 绑定事件
        self.browse_tif_button.clicked.connect(self.browse_tif_button_click)
        self.browse_folder_button.clicked.connect(self.browse_folder_button_click)
        self.reset_button.clicked.connect(self.reset_value)
        self.start_button.clicked.connect(self.start_processing)
        self.save_button.clicked.connect(self.save_result)

    def handle_error(self, e):
        QMessageBox.critical(self, "ERROR", self.tr(str(e)))
        exit(1)

    def handle_warning(self, w):
        for each in w:
            QMessageBox.warning(self, "WARNING", self.tr(str(each.message)))

    def save_result(self):
        if self.busy or self.image_mat is None:
            return
        im = Image.fromarray(self.result_mat)
        filepath, _ = QFileDialog.getSaveFileName(self, "Save result", None, "TIF file(*.tif *.tiff)")
        if filepath != "":
            try:
                im.save(filepath)
            except IOError:
                QMessageBox.warning(self, "Failure", self.tr("Failed to write to file!"))
            else:
                QMessageBox.information(self, "Successful", self.tr("Result has been saved"))

    def start_processing(self):
        if self.image_mat is None:
            QMessageBox.information(self, "No Image", self.tr("Please select image file!"))
            return

        if self.busy:
            return

        kwargs = {
            'images': self.image_mat,
            'mag': self.mag_spin_box.value(),
            'radius': self.rad_spin_box.value(),
            'sample_n': self.axes_spin_box.value(),
            'do_gw': self.gw_checkbox.checkState() != 0,
            'do_iw': self.iw_checkbox.checkState() != 0,
            'sofi_delay': self.trac_delay_spin_box.value(),
            'sofi_order': 0 if self.tra_radio_button.isChecked() else self.trac_order_spin_box.value(),
            'avg_every_n': self.n_frame_spin_box.value(),
            'do_drift_correction': self.dc_checkbox.checkState() != 0,
        }
        self.progress_thread = ProgressBarThread(srrf, kwargs)
        self.progress_thread.progress_update.connect(self.progressBar.setValue)

        def ret_result(value):
            self.result_mat = value
            self.pyqtimv_2.setImage(self.result_mat, axes={'x': 1, 'y': 0})
            std = float(np.std(self.result_mat))
            self.pyqtimv_2.setLevels(0, 2 * std)
            self.save_button.setEnabled(True)
            self.busy = False

        self.progress_thread.result.connect(ret_result)
        self.progress_thread.error_signal.connect(self.handle_error)
        self.progress_thread.warning_signal.connect(self.handle_warning)
        self.busy = True
        self.progress_thread.start()

    def reset_value(self):
        self.mag_spin_box.setValue(self.default_values[0])
        self.rad_spin_box.setValue(self.default_values[1])
        self.axes_spin_box.setValue(self.default_values[2])
        self.n_frame_spin_box.setValue(self.default_values[3])
        self.dc_checkbox.setCheckState(self.default_values[4])
        self.iw_checkbox.setCheckState(self.default_values[5])
        self.gw_checkbox.setCheckState(self.default_values[6])
        self.tra_radio_button.setChecked(self.default_values[7])
        self.trac_radio_button.setChecked(self.default_values[8])
        self.trac_order_spin_box.setValue(self.default_values[9])
        self.trac_delay_spin_box.setValue(self.default_values[10])
        self.trac_order_spin_box.setEnabled(True)
        self.trac_delay_spin_box.setEnabled(True)

    def browse_tif_button_click(self):
        if self.busy:
            return
        file_name, _ = QFileDialog.getOpenFileName(QWidget(), 'Open TIF File', None, "TIF file(*.tif *.tiff)")
        if file_name == "":
            return
        if os.path.splitext(file_name)[-1] not in ('.tif', '.tiff'):
            QMessageBox.information(self, "Information", self.tr("Please select a TIF file!"))
        else:
            self.lineEdit.setText(file_name)
            self.progress_thread = ProgressBarThread(read_images_from_tiff_file, file_name)
            self.progress_thread.progress_update.connect(self.progressBar.setValue)

            def ret_result(value):
                if value is not None:
                    self.image_mat = value
                    self.pyqtimv.setImage(self.image_mat, axes={'t': 0, 'x': 2, 'y': 1}, autoLevels=True)
                    self.pyqtimv.play(20)
                    self.start_button.setEnabled(True)
                self.busy = False

            self.progress_thread.result.connect(ret_result)
            self.progress_thread.error_signal.connect(self.handle_error)
            self.progress_thread.warning_signal.connect(self.handle_warning)
            self.busy = True
            self.progress_thread.start()

    def browse_folder_button_click(self):
        if self.busy:
            return
        folder_path = QFileDialog.getExistingDirectory(QWidget(), 'Open folder', None)
        if folder_path == "":
            return
        self.lineEdit.setText(folder_path)
        self.progress_thread = ProgressBarThread(read_images_from_folder, folder_path)
        self.progress_thread.progress_update.connect(self.progressBar.setValue)

        def ret_result(value):
            if value is not None:
                self.image_mat = value
                self.pyqtimv.setImage(self.image_mat, axes={'t': 0, 'x': 2, 'y': 1}, autoLevels=True)
                self.pyqtimv.play(20)
                self.start_button.setEnabled(True)
            self.busy = False

        self.progress_thread.result.connect(ret_result)
        self.progress_thread.error_signal.connect(self.handle_error)
        self.progress_thread.warning_signal.connect(self.handle_warning)
        self.busy = True
        self.progress_thread.start()


def main():
    app = QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
