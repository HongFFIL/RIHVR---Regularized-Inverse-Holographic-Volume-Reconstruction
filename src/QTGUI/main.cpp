#include "rihvr.h"

#include <QApplication>
#include <QLocale>
#include <QTranslator>


#include <QApplication>
#include <QLabel>
#include <QMovie>
#include <QTimer>




int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QTranslator translator;
    const QStringList uiLanguages = QLocale::system().uiLanguages();
    for (const QString &locale : uiLanguages) {
        const QString baseName = "RIHVR_GUI_" + QLocale(locale).name();
        if (translator.load(":/i18n/" + baseName)) {
            a.installTranslator(&translator);
            break;
        }
    }

    // create the main RIHVR window
    RIHVR w;

    // set app icon
    w.setWindowIcon(QIcon(":/Icons/MenuBar/Icons/RIHVR_Icon.ico"));

    // add a splash screen gif
    // Create a label that will act as splash screen
    QLabel splash;
    splash.setWindowFlags(Qt::SplashScreen | Qt::FramelessWindowHint);
    splash.setAttribute(Qt::WA_TranslucentBackground);

    // Load animated gif
    QMovie movie(":/Icons/MenuBar/Icons/Splash_3.gif");
    splash.setMovie(&movie);
    movie.start();

    // Update size to match the gif
    splash.adjustSize();

    // Center the splash screen
    QScreen *screen = QGuiApplication::primaryScreen();
    QRect screenGeometry = screen->geometry();
    int x = (screenGeometry.width() - splash.width()) / 2;
    int y = (screenGeometry.height() - splash.height()) / 2;
    splash.move(x, y);

    splash.show();

    QTimer::singleShot(3000, [&]() {
        splash.close();
        w.show();
    });


    // ---- Global Dark Theme ----
    a.setStyleSheet(
        "QCheckBox {"
        "    color: #dddddd;"                  /* Text color */
        "    spacing: 5px;"                    /* Space between indicator and text */
        "}"

        "QCheckBox::indicator {"
        "    width: 16px;"
        "    height: 16px;"
        "    border-radius: 3px;"              /* Slightly rounded square */
        "    border: 1px solid #ffffff;"       /* Light outline */
        "    background-color: transparent;"
        "}"

        "QCheckBox::indicator:hover {"
        "    border: 1px solid #00ccff;"       /* Highlight border on hover */
        "}"

        /* Checked box with a blue checkmark */
        "QCheckBox::indicator:checked {"
        "    background-color: #00aaff;"       /* Bright blue background */
        "    border: 1px solid #00aaff;"
        "}"

        "QCheckBox::indicator:indeterminate {"
        "    background-color: #0077aa;"       /* Slightly darker for mixed state */
        "    border: 1px solid #00aaff;"
        "}"
        "QRadioButton {"
        "   color: #dddddd;"                  /* Text color */
        "   spacing: 5px;"                     /* Space between indicator and text */
        "}"

        "QRadioButton::indicator {"
        "   width: 16px;"
        "   height: 16px;"
        "   border-radius: 8px;"               /* Make circle perfectly round */
        "   border: 1px solid #ffffff;"        /* Light outline */
        "   background-color: transparent;"
        "}"

        "QRadioButton::indicator:checked {"
        "   background-color: #00aaff;"        /* Bright blue when selected */
        "   border: 1px solid #00aaff;"
        "}"

        "QRadioButton::indicator:hover {"
        "   border: 1px solid #00ccff;"        /* Slight highlight on hover */
        "}"
        "QMainWindow, QWidget {"
        "   background-color: #1e1e1e;"
        "   color: #dddddd;"
        "   font-family: 'Segoe UI';"
        "   font-size: 10pt;"
        "}"

        "QMenuBar {"
        "   background-color: #2d2d2d;"
        "   color: #dddddd;"
        "}"

        "QMenuBar::item:selected {"
        "   background-color: #3a3a3a;"
        "}"

        "QMenu {"
        "   background-color: #2d2d2d;"
        "   color: #dddddd;"
        "}"

        "QMenu::item:selected {"
        "   background-color: #3a3a3a;"
        "}"

        "QTabWidget::pane {"
        "   border: 1px solid #3a3a3a;"
        "   background-color: #1e1e1e;"
        "}"

        "QTabBar::tab {"
        "   background: #2d2d2d;"
        "   color: #cccccc;"
        "   padding: 8px 20px;"
        "   border: 1px solid #3a3a3a;"
        "   border-bottom: none;"
        "}"

        "QTabBar::tab:selected {"
        "   background: #3a3a3a;"
        "   color: white;"
        "}"

        "QPushButton {"
        "   background-color: #2d2d2d;"
        "   color: #ffffff;"
        "   border: 1px solid #3a3a3a;"
        "   border-radius: 4px;"
        "   padding: 5px;"
        "}"

        "QPushButton:hover {"
        "   background-color: #3a3a3a;"
        "}"

        "QLineEdit, QTextEdit, QPlainTextEdit {"
        "   background-color: #252525;"
        "   color: #ffffff;"
        "   border: 1px solid #3a3a3a;"
        "   border-radius: 4px;"
        "}"

        "QComboBox, QSpinBox, QDoubleSpinBox {"
        "   background-color: #252525;"
        "   color: #ffffff;"
        "   border: 1px solid #3a3a3a;"
        "   border-radius: 4px;"
        "}"

        "QStatusBar {"
        "   background-color: #2d2d2d;"
        "   color: #aaaaaa;"
        "}"
        );
    // ----------------------------


    return a.exec();
}
