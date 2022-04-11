import random
import sys
import time
from threading import Thread

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QApplication, QWidget, QPlainTextEdit, QPushButton, QLabel, QRadioButton, QDialog, QCheckBox
from qt_material import apply_stylesheet

import predict


def sss(widget, size=20, style=0, font="default"):  # setStyleSheet
    css = f"font-size:{size}px;"
    if style:
        if style == 1:
            css += "font-weight:bold;"
        else:
            css += "font-weight:light;"
    if font == 'default':
        css += "font-family:JetBrains Mono;"
    else:
        css += f"font-family:{font};"
    widget.setStyleSheet(css)


class Init(QDialog):
    def __init__(self):
        super(Init, self).__init__()
        self.setFixedSize(600, 200)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        self._time = 0
        self.loading_plate = QLabel(self)
        self.loading = QMovie('loading.gif')
        self.loading_plate.setGeometry(-35, -20, 200, 200)
        self.loading.setScaledSize(QSize(200, 200))
        self.loading_plate.setMovie(self.loading)
        self.loading.start()

        do_list_name = ['Generating BERT Tokenizer',
                        'Loading Count Vectoizer',
                        'Loading TF-IDF Transformer',
                        'Building Models (0/4)',
                        'Testing Core']
        self.do_list = []
        for _ in range(4):
            __ = QCheckBox(do_list_name[_], self)
            __.setGeometry(130, 50 * _, 465, 50)
            sss(__)
            self.do_list.append(__)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_status)
        self.timer.start(1000)

        self.load = QTimer(self)
        self.load.singleShot(10, self.init)

        self.layer = QLabel(self)
        self.layer.setGeometry(130, 0, 200, 200)

        self.btn = QPushButton('START', self)
        self.btn.setEnabled(False)
        self.btn.setGeometry(25, 145, 80, 40)
        sss(self.btn, 15)
        self.btn.clicked.connect(self.start)

    def closeEvent(self, event):
        exit(0)

    def init(self):
        thread1 = Thread(target=self.load_tokenizer)
        thread1.start()
        thread2 = Thread(target=self.load_cntizer)
        thread2.start()
        thread3 = Thread(target=self.load_tfizer)
        thread3.start()
        thread4 = Thread(target=self.load_model)
        thread4.start()

    def load_tokenizer(self):
        global tokenizer
        tokenizer = predict.Initialize().load_tokenizer()
        self.do_list[0].setChecked(True)
        self.do_list[0].setText('Generating BERT Tokenizer (Done)')
        print(0)

    def load_cntizer(self):
        global cntizer
        cntizer = predict.Initialize().load_count_vectorizer()
        self.do_list[1].setChecked(True)
        self.do_list[1].setText('Loading Count Vectoizer (Done)')
        print(1)

    def load_tfizer(self):
        global tfizer
        tfizer = predict.Initialize().load_tfidf_transformer()
        self.do_list[2].setChecked(True)
        self.do_list[2].setText('Loading TF-IDF Transformer (Done)')
        print(2)

    def load_model(self):
        global model
        for _ in range(4):
            model.append(predict.Initialize().load_model(_))
            self.do_list[3].setText(f'Building Models ({_ + 1}/4)')
            print(3)
        self.do_list[3].setChecked(True)

    def check_status(self):
        flag = True
        for _ in range(4):
            if not self.do_list[_].isChecked():
                flag = False
                break
        if flag:
            self.loading.stop()
            self.btn.setEnabled(True)

    def test_core(self):
        pre_test = predict.PredictWithBERT('', tokenizer, model[0]).predict()
        self.do_list[4].setText(f'Testing Core (Done)')
        self.do_list[4].setChecked(True)
        print(4)
        self.loading.stop()
        self.btn.setEnabled(True)

    def start(self):
        self.timer.stop()
        demo.hide()
        main_ui.show()


class Test(QWidget):
    def __init__(self):
        super(Test, self).__init__()
        self.setWindowTitle('MBTI Personality Predictor')

        self.MainWidget = QWidget(self)
        self.setFixedSize(1210, 850)

        title = QLabel('MBTI Personality Predictor', self.MainWidget)
        title.move(25, 20)
        sss(title, 30, 1)

        self.textEdit = QPlainTextEdit(self.MainWidget)
        self.textEdit.setPlaceholderText("Input something here to analyze")
        self.textEdit.move(25, 90)
        self.textEdit.resize(550, 575)
        sss(self.textEdit)

        btn_line_temp = QPushButton(self.MainWidget)
        btn_line_temp.setGeometry(20, 70, 700, 2)

        _ = []
        for _i in range(3):
            __ = QLabel(self.MainWidget)
            __.setGeometry(650, 262 + 180 * _i, 500, 1)
            __.setStyleSheet('background:#202224;')
            _.append(__)

        self.btn_confirm = QPushButton('Confirm', self.MainWidget)
        self.btn_confirm.resize(150, 50)
        self.btn_confirm.move(25, 690)
        sss(self.btn_confirm)
        self.btn_confirm.clicked.connect(self.confirm)

        self.btn_cancel = QPushButton('Clear', self.MainWidget)
        self.btn_cancel.resize(150, 50)
        self.btn_cancel.move(190, 690)
        sss(self.btn_cancel)
        self.btn_cancel.clicked.connect(self.clear)

        self.result = QLabel(self.MainWidget)
        self.result.resize(550, 50)
        self.result.move(350, 740)
        sss(self.result)

        _keyword = [['Introverted', 'Extraverted'],
                    ['Intuitive', 'Sensitive'],
                    ['Feeling', 'Thinking'],
                    ['Judging', 'Prospecting']]
        self.value = []
        self.keyword = []
        for _i in range(4):
            self.value.append([])
            self.keyword.append([])
            for _j in range(2):
                keyword = QLabel(self.MainWidget)
                keyword.setText(_keyword[_i][_j])
                keyword.move(1030 if _j else 620, 210 + 185 * _i)
                keyword.resize(150, 40)
                sss(keyword)
                value = QLabel(self.MainWidget)
                value.setText('N/A')
                value.move(1100 if _j else 600, 165 + 185 * _i)
                value.resize(100, 40)
                sss(value)
                value.setAlignment(Qt.AlignCenter)
                keyword.setAlignment(Qt.AlignRight if _j else Qt.AlignLeft)
                self.value[_i].append(value)
                self.keyword[_i].append(keyword)

        _label = ['Mind', 'Energy', 'Nature', 'Tactics']
        self.label = []
        for _i in range(4):
            label = QLabel(self.MainWidget)
            label.setText(_label[_i])
            label.resize(500, 100)
            label.setAlignment(Qt.AlignCenter)
            sss(label, 28, 1)
            label.move(660, 60 + 185 * _i)
            self.label.append(label)

        _desc = ['This trait determines how we interact with our environment.',
                 'This trait shows where we direct our mental energy.',
                 'This trait determines how we make decisions and cope with emotions.',
                 'This trait reflects our approach to work, planning and decision-making.']
        self.desc = []
        for _i in range(4):
            desc = QLabel(self.MainWidget)
            desc.setText(_desc[_i])
            desc.resize(600, 100)
            desc.setAlignment(Qt.AlignCenter)
            desc.setWordWrap(True)
            sss(desc, 14)
            desc.move(600, 95 + 185 * _i)
            self.desc.append(desc)

        self.bar = []
        for _i in range(4):
            bar = QLabel(self.MainWidget)
            bar.resize(400, 10)
            bar.setStyleSheet("background:#1e2124")
            bar.move(700, 180 + 185 * _i)

        color = ['#4499bb', '#ddbb33', '#33aa77', '#886699']
        self.res_bar = []
        for _i in range(4):
            bar = QLabel(self.MainWidget)
            bar.resize(0, 0)
            bar.setStyleSheet(f"background:{color[_i]}")
            bar.move(700, 180 + 185 * _i)
            self.res_bar.append(bar)

        self.method1 = QRadioButton('TF-IDF + XGBoost (Faster)', self.MainWidget)
        self.method2 = QRadioButton('BERT + Tensorflow (Slower)', self.MainWidget)
        self.method1.setGeometry(25, 755, 265, 35)
        self.method2.setGeometry(25, 795, 265, 35)
        sss(self.method1, 15)
        sss(self.method2, 15)
        self.method1.setChecked(True)

    def confirm(self):
        color = ['#4499bb', '#ddbb33', '#33aa77', '#886699']
        type_chart, result = predict.Predict(self.textEdit.toPlainText(), cntizer, tfizer).predict() if self.method1.isChecked() else predict.PredictWithBERT(self.textEdit.toPlainText(), tokenizer, model).predict()
        text = 'The result is: '
        _ = ['IE', 'NS', 'FT', 'JP']
        for _i in range(4):
            text += _[_i][0] if type_chart[_i] else _[_i][1]
        self.result.setText(text)
        for _i in range(4):
            value = result[_i]
            self.value[_i][0].setText(f'{str(value)[:4]}%')
            self.value[_i][1].setText(f'{str(100.1 - value)[:4]}%')
        for _i in range(4):
            if result[_i] > 50:
                self.res_bar[_i].resize(int(4 * result[_i]), 10)
                self.res_bar[_i].move(700, 180 + 185 * _i)
                self.value[_i][0].setStyleSheet(f'color:{color[_i]};'
                                                f'font-family:JetBrains Mono;'
                                                f'font-weight:bold;'
                                                f'font-size:20px;')
                self.value[_i][1].setStyleSheet(f'color:#ffffff;'
                                                f'font-family:JetBrains Mono;'
                                                f'font-weight:normal;'
                                                f'font-size:20px;')
                self.keyword[_i][0].setStyleSheet(f'color:{color[_i]};'
                                                  f'font-family:JetBrains Mono;'
                                                  f'font-weight:bold;'
                                                  f'font-size:20px;')
                self.keyword[_i][1].setStyleSheet(f'color:#ffffff;'
                                                  f'font-family:JetBrains Mono;'
                                                  f'font-weight:normal;'
                                                  f'font-size:20px;')
            else:
                self.res_bar[_i].resize(int(4 * (100 - result[_i])), 10)
                self.res_bar[_i].move(int(1100 - 4 * (100 - result[_i])), 180 + 185 * _i)
                self.value[_i][1].setStyleSheet(f'color:{color[_i]};'
                                                f'font-family:JetBrains Mono;'
                                                f'font-weight:bold;'
                                                f'font-size:20px;')
                self.value[_i][0].setStyleSheet(f'color:#ffffff;'
                                                f'font-family:JetBrains Mono;'
                                                f'font-weight:normal;'
                                                f'font-size:20px;')
                self.keyword[_i][1].setStyleSheet(f'color:{color[_i]};'
                                                  f'font-family:JetBrains Mono;'
                                                  f'font-weight:bold;'
                                                  f'font-size:20px;')
                self.keyword[_i][0].setStyleSheet(f'color:#ffffff;'
                                                  f'font-family:JetBrains Mono;'
                                                  f'font-weight:normal;'
                                                  f'font-size:20px;')

    def clear(self):
        self.textEdit.setPlainText('')
        self.result.setText('')
        for _i in range(4):
            self.res_bar[_i].setGeometry(0, 0, 0, 0)
            for _j in range(2):
                self.value[_i][_j].setText('N/A')
                self.value[_i][_j].setStyleSheet(f'color:#ffffff;'
                                                 f'font-family:JetBrains Mono;'
                                                 f'font-weight:normal;'
                                                 f'font-size:20px;')
                self.keyword[_i][_j].setStyleSheet(f'color:#ffffff;'
                                                   f'font-family:JetBrains Mono;'
                                                   f'font-weight:normal;'
                                                   f'font-size:20px;')


if __name__ == '__main__':
    tokenizer = None
    cntizer = None
    tfizer = None
    model = []
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    demo = Init()
    main_ui = Test()
    demo.open()
    sys.exit(app.exec_())
