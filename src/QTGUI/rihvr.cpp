#include "rihvr.h"
#include "./ui_rihvr.h"
// for the about message box
#include <QMessageBox>
// including the settings dialog box we created
#include "settingsdialog.h"
// for the file browser
#include <QFileDialog>
#include <QSettings>
#include <QRegularExpression>
#include <QIntValidator>
#include <QDoubleValidator>
// RIHVR startup file
#include <QFile>
#include <QTextStream>
#include <QStandardPaths>
#include <QDebug>
// for saving JSON files
#include <QJsonObject>
#include <QJsonDocument>
#include <QFile>
#include <QJsonArray>
#include <algorithm>

// the tracking plot module
#include "tracksplot.h"

// for the visualizaiton utilities
#include "visualizationutils.h"

// for the bgcompute functions
#include "bgcompute.h"

// for the tracking functions
#include "trackingcompute.h"

// for running processes
#include <QProcessEnvironment>
#include <QProcess>
#include <QTimer>

// for the sparse reconstricon functions
#include "reconstructioncompute.h"

// for the settings dialog box
#include "settingsdialog.h"

// scroll bar for the console
#include <QScrollBar>

// to use images inside QT
#include <QImage>
#include <QImageReader>

RIHVR::RIHVR(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::RIHVR)
{
    ui->setupUi(this);

    // default window size
    this->resize(940, 750);     // default width x height
    this->setMinimumSize(940, 750); // optional: prevent shrinking too small

    // start from the Background tab
    ui->tabWidget->setCurrentIndex(0);

    // in RIHVR.cpp constructor
    settingsdialog = new SettingsDialog(this);

    // ================================================================== Console Page ==================================================================//
    //============================ stop button
    // Load both icons from resources
    QIcon stopIconNormal(":/Icons/MenuBar/Icons/stopButton.svg");
    QIcon stopIconPressed(":/Icons/MenuBar/Icons/stopButtonPressed.svg");

    // Configure button
    ui->StopButton->setIcon(stopIconNormal);
    ui->StopButton->setIconSize(QSize(64, 64)); // adjust to desired size
    ui->StopButton->setFlat(true);               // removes button borders
    ui->StopButton->setFixedSize(64, 64);       // button slightly bigger than icon
    ui->StopButton->setText("");                 // ensure no text

    // Swap icon on press/release
    connect(ui->StopButton, &QPushButton::pressed, this, [=]() {
        ui->StopButton->setIcon(stopIconPressed);
    });
    connect(ui->StopButton, &QPushButton::released, this, [=]() {
        ui->StopButton->setIcon(stopIconNormal);
    });

    //============================ clear button
    // Load both icons from resources
    QIcon clearIconNormal(":/Icons/MenuBar/Icons/clearButton.svg");
    QIcon clearIconPressed(":/Icons/MenuBar/Icons/clearButtonPressed.svg");

    // Configure button
    ui->ClearButton->setIcon(clearIconNormal);
    ui->ClearButton->setIconSize(QSize(64, 64)); // adjust to desired size
    ui->ClearButton->setFlat(true);               // removes button borders
    ui->ClearButton->setFixedSize(64, 64);       // button slightly bigger than icon
    ui->ClearButton->setText("");                 // ensure no text

    // Swap icon on press/release
    connect(ui->ClearButton, &QPushButton::pressed, this, [=]() {
        ui->ClearButton->setIcon(clearIconPressed);
    });
    connect(ui->ClearButton, &QPushButton::released, this, [=]() {
        ui->ClearButton->setIcon(clearIconNormal);
    });

    //============================ log button
    // Load both icons from resources
    QIcon logIconNormal(":/Icons/MenuBar/Icons/logButton.svg");
    QIcon logIconPressed(":/Icons/MenuBar/Icons/logButtonPressed.svg");

    // Configure button
    ui->SaveLogButton->setIcon(logIconNormal);
    ui->SaveLogButton->setIconSize(QSize(64, 64)); // adjust to desired size
    ui->SaveLogButton->setFlat(true);               // removes button borders
    ui->SaveLogButton->setFixedSize(64, 64);       // button slightly bigger than icon
    ui->SaveLogButton->setText("");                 // ensure no text

    // Swap icon on press/release
    connect(ui->SaveLogButton, &QPushButton::pressed, this, [=]() {
        ui->SaveLogButton->setIcon(logIconPressed);
    });
    connect(ui->SaveLogButton, &QPushButton::released, this, [=]() {
        ui->SaveLogButton->setIcon(logIconNormal);
    });

    // connect the stop button
    connect(ui->StopButton, &QPushButton::clicked, this, &RIHVR::stopAllProcesses);

    // connect the clear console button
    connect(ui->ClearButton, &QPushButton::clicked, this, &RIHVR::clearConsole);

    // connect the save console log button
    connect(ui->SaveLogButton, &QPushButton::clicked, this, &RIHVR::saveLogConsole);

    // ================================================================== Menu Bar ==================================================================//
    // Connect Exit menu button item to close the application
    connect(ui->actionExit, &QAction::triggered, this, &QMainWindow::close);

    // Connect About meny button to the about dialog box
    connect(ui->actionAbout, &QAction::triggered, this, &RIHVR::showAboutDialog);

    // Connect About meny button to the about dialog box
    connect(ui->actionLicense, &QAction::triggered, this, &RIHVR::showLicenseDialog);

    // Connect the load and save state menu buttons
    connect(ui->actionSave_State, &QAction::triggered, this, &RIHVR::saveStateToFile);
    connect(ui->actionLoad_State, &QAction::triggered, this, [this]() {
        loadStateFromFile();  // default argument used, opens dialog
    });

    // setting dialog box
    connect(ui->actionSettings, &QAction::triggered, this, [this]() {
        if (settingsdialog->exec() == QDialog::Accepted) {
            // The dialog already saved its own state internally, so now just update any main-window variables
            currentDefaultRihvrPath = settingsdialog->getDefaultJsonPath();
        }
    });
    // ensure that we have the settings dialog running
    settingsdialog = new SettingsDialog(this);  // done once

    // ================================================================== Background Page ==================================================================//

    // RIHVR logo
    QPixmap pix(":/Icons/MenuBar/Icons/Logo.png");
    pix = pix.scaled(ui->RIHVRLabel->size(),
                     Qt::KeepAspectRatio,
                     Qt::SmoothTransformation);
    ui->RIHVRLabel->setPixmap(pix);

    // ensure that the numeric fields only accept numbers
    ui->StartImage->setValidator(new QIntValidator(0, 9999999, this));
    ui->EndImage->setValidator(new QIntValidator(0, 9999999, this));
    ui->BGStart->setValidator(new QIntValidator(0, 9999999, this));
    ui->BGEnd->setValidator(new QIntValidator(0, 9999999, this));
    ui->WindowSize->setValidator(new QIntValidator(0, 9999999, this));
    ui->ResizePercent->setValidator(new QDoubleValidator(0.0, 9999999.0, 3, this));
    ui->MinFrames->setValidator(new QIntValidator(0, 9999999, this));
    ui->MaxFrames->setValidator(new QIntValidator(0, 9999999, this));
    // Initial text
    ui->ImgPath->setPlaceholderText("Select an image from the image folder.");
    ui->ImgFileFormat->setPlaceholderText("In C-style formatting (e.g. data_%04.tif). Fills automatically when a sample input image is selected.");
    ui->BGpath->setPlaceholderText("Select the directory for storing computed background(s).");
    ui->StartImage->setPlaceholderText("Default fills automatically once Image Path is selected.");
    ui->EndImage->setPlaceholderText("Default fills automatically once Image Path is selected.");
    ui->BGStart->setPlaceholderText("Default is image start number.");
    ui->BGEnd->setPlaceholderText("Default is image end number.");
    ui->WindowSize->setPlaceholderText("# of frames for moving average.");
    ui->MinFrames->setPlaceholderText("Minimum number of frames considered to generate BG.");
    ui->MaxFrames->setPlaceholderText("Maximum number of frames considered to generate BG.");

    // Image path (from the sample image, we can extract the number of images, image prefix, etc. to populate the related fields)
    connect(ui->ImgPathBrowse, &QPushButton::clicked, this, [this]() {
        QString sampleFile = QFileDialog::getOpenFileName(
            this,
            tr("Select Sample Image"),
            ui->ImgPath->text(),
            tr("Images (*.tif *.jpg *.png *.bmp *.tiff)")
            );

        if (sampleFile.isEmpty())
            return;

        QFileInfo fileInfo(sampleFile);
        QString folderPath = fileInfo.absolutePath();
        QString fileName = fileInfo.fileName();   // e.g. "Im_000004.jpg"
        QString suffix = fileInfo.suffix();       // e.g. "jpg"

        // Ensure the folder path ends with a separator (RIHVR expects '/' or '\' in the very end of the path)
        // Always convert to forward slashes
        folderPath.replace('\\', '/');
        // Ensure it ends with a forward slash
        if (!folderPath.endsWith('/'))
            folderPath += '/';

        // Match the LAST numeric group in the filename
        QRegularExpression lastNumRx("(\\d+)(?!.*\\d)");
        QRegularExpressionMatch m = lastNumRx.match(fileName);
        if (!m.hasMatch()) {
            QMessageBox::warning(this, tr("Error"), tr("No numeric part found in file name."));
            return;
        }

        QString numberStr = m.captured(1);        // e.g. "000004"
        int numDigits = numberStr.length();       // e.g. 6
        int startIndex = m.capturedStart(1);      // index of that numeric group in fileName

        // prefix = part before the numeric section, suffix (extension) already known
        QString prefix = fileName.left(startIndex);                // e.g. "Im_"
        QString extension = suffix;                                // e.g. "jpg"

        // Build the printf-style pattern safely: e.g. "Im_%06d.jpg"
        QString filePattern = prefix + "%" + QString("0%1d").arg(numDigits) + "." + extension;

        // Now scan the directory for files with same prefix and extension to find min/max index.
        QDir dir(folderPath);
        // Build a name filter like "Im_*.jpg"
        QString nameFilter = prefix + "*" + "." + extension;
        QStringList nameFilters;
        nameFilters << nameFilter;
        dir.setNameFilters(nameFilters);

        QStringList fileList = dir.entryList(QDir::Files, QDir::Name);
        if (fileList.isEmpty()) {
            // still update UI with the single-file info
            ui->ImgPath->setText(folderPath);
            ui->ImgFileFormat->setText(filePattern);
            ui->StartImage->setText(QString::number(numberStr.toInt()));
            ui->EndImage->setText(QString::number(numberStr.toInt()));
            return;
        }

        int minNum = numberStr.toInt();
        int maxNum = numberStr.toInt();

        // Use regex to extract the last numeric group in each filename
        QRegularExpression numRx("(\\d+)(?!.*\\d)");
        for (const QString &fn : fileList) {
            QRegularExpressionMatch mm = numRx.match(fn);
            if (mm.hasMatch()) {
                int n = mm.captured(1).toInt();
                if (n < minNum) minNum = n;
                if (n > maxNum) maxNum = n;
            }
        }
    //

    // Update UI fields in the background tab according to what we figured out from the sample image path
    ui->ImgPath->setText(folderPath);
    ui->ImgFileFormat->setText(filePattern);
    ui->StartImage->setText(QString::number(minNum));
    ui->EndImage->setText(QString::number(maxNum));

    // also set this as the max frames
    ui->MaxFrames->setText(QString::number(maxNum));

    ui->BGStart->setText(QString::number(minNum));
    ui->BGEnd->setText(QString::number(maxNum));
    });

    // BG path -- also ensures that the path name ends with a slash
    connect(ui->BGPathBrowse, &QPushButton::clicked, this, [this]() {
        QString dir = QFileDialog::getExistingDirectory(this, tr("Select Output Folder"), ui->BGpath->text());
        if (!dir.isEmpty()) {
            // Normalize path: forward slashes and trailing slash
            dir.replace('\\', '/');  // replace backslashes with forward slashes
            if (!dir.endsWith('/'))  // ensure trailing slash
                dir += '/';

            ui->BGpath->setText(dir);
        }
    });

    // Connect compute background radio buttons
    connect(ui->YesBackgroundCompute, &QRadioButton::toggled, this, &RIHVR::updateBackgroundTab);
    connect(ui->NoBackgroundCompute, &QRadioButton::toggled, this, &RIHVR::updateBackgroundTab);

    // Connect BG type raido buttons
    connect(ui->MeanBG, &QRadioButton::toggled, this, &RIHVR::updateBackgroundTab);
    connect(ui->MedianBG, &QRadioButton::toggled, this, &RIHVR::updateBackgroundTab);
    connect(ui->MovingBG, &QRadioButton::toggled, this, &RIHVR::updateBackgroundTab);
    connect(ui->CCBG, &QRadioButton::toggled, this, &RIHVR::updateBackgroundTab);

    // Line edits (check for text changes) -- to enable the compute BG button
    QList<QLineEdit*> lineEdits = {
        ui->ImgPath, ui->BGpath, ui->StartImage, ui->EndImage,
        ui->BGStart, ui->BGEnd, ui->ImgFileFormat
    };

    for (QLineEdit* le : lineEdits)
        connect(le, &QLineEdit::textChanged, this, &RIHVR::updateBackgroundTab);

    // Moving BG options line edits
    connect(ui->WindowSize, &QLineEdit::textChanged, this, &RIHVR::updateBackgroundTab);

    // Initial state
    updateBackgroundTab();

    //============ Compute background button
    // different functions for moving and other BGTypes
    connect(ui->ComputeBGButton, &QPushButton::clicked, this, [this]() {
        if (ui->MovingBG->isChecked()) {
            computeMovingBackground();
        } else if (ui->CCBG->isChecked()){
            computeCCBackground();
        }else {
            computeBackground();
        }
    });

    // coupling signal between the shared fields
    static bool syncing = false; // prevent recursive signals when syncing

    // When ImgPath changes, update HoloImgPath
    connect(ui->ImgPath, &QLineEdit::textChanged, this, [this](const QString &text) {
        static bool localSync = false;
        if (localSync) return; // prevent loop
        localSync = true;
        imgPathShared = text;
        ui->HoloImgPath->setText(text);
        localSync = false;
    });

    // When ImgFileFormat changes, update HoloImgFileFormat
    connect(ui->ImgFileFormat, &QLineEdit::textChanged, this, [this](const QString &text) {
        static bool localSync = false;
        if (localSync) return; // prevent loop
        localSync = true;
        imgFileFormatShared = text;
        ui->HoloImgFileFormat->setText(text);
        localSync = false;
    });

    // When StartImg changes, update HoloStartImg
    connect(ui->StartImage, &QLineEdit::textChanged, this, [this](const QString &text) {
        static bool localSync = false;
        if (localSync) return; // prevent loop
        localSync = true;
        imgStartIdXShared = text;
        ui->HoloStartImage->setText(text);
        localSync = false;
    });

    // When EndImg changes, update HoloEndImg
    connect(ui->EndImage, &QLineEdit::textChanged, this, [this](const QString &text) {
        static bool localSync = false;
        if (localSync) return; // prevent loop
        localSync = true;
        imgEndIdXShared = text;
        ui->HoloEndImg->setText(text);
        localSync = false;
    });

    // When BGpath changes, update HoloBGpath
    connect(ui->BGpath, &QLineEdit::textChanged, this, [this](const QString &text) {
        static bool localSync = false;
        if (localSync) return; // prevent loop
        localSync = true;
        bgPathShared = text;
        ui->HoloBGpath->setText(text);
        localSync = false;
    });

    // if no background is computed, default to none background in the processing page
    connect(ui->NoBackgroundCompute, &QCheckBox::toggled, this, [this](bool checked){
        if (checked) {
            // Automatically select "NoBG" on the Processing tab
            ui->ReconNoBG->setChecked(true);
        }
    });

    // if mean bg is computed, default to mean background in the processing page
    connect(ui->MeanBG, &QCheckBox::toggled, this, [this](bool checked){
        if (checked) {
            // Automatically select "MeanBG" on the Processing tab
            ui->ReconMeanBG->setChecked(true);
        }
    });

    // if median bg is computed, default to median background in the processing page
    connect(ui->MedianBG, &QCheckBox::toggled, this, [this](bool checked){
        if (checked) {
            // Automatically select "MedianBG" on the Processing tab
            ui->ReconMedianBG->setChecked(true);
        }
    });

    // if moving bg is computed, default to moving background in the processing page
    connect(ui->MovingBG, &QCheckBox::toggled, this, [this](bool checked){
        if (checked) {
            // Automatically select "MedianBG" on the Processing tab
            ui->ReconMovingBG->setChecked(true);
        }
    });

    // if correlation bg is computed, default to correlation background in the processing page
    connect(ui->CCBG, &QCheckBox::toggled, this, [this](bool checked){
        if (checked) {
            // Automatically select "MeanBG" on the Processing tab
            ui->ReconCorrelationBG->setChecked(true);
        }
    });


    // ================================================================== Hologram reconstruction page ==================================================================//
    // ensure that the numeric fields only accept numbers
    ui->HoloStartImage->setValidator(new QIntValidator(0, 9999999, this));
    ui->HoloEndImg->setValidator(new QIntValidator(0, 9999999, this));
    ui->HololIncrement->setValidator(new QIntValidator(0, 9999999, this));
    ui->Nz->setValidator(new QIntValidator(1, 9999999, this));
    ui->ZeroPad->setValidator(new QIntValidator(0, 9999999, this));
    ui->nFISTA->setValidator(new QIntValidator(1, 9999999, this));
    ui->nTV->setValidator(new QIntValidator(1, 9999999, this));
    ui->x0->setValidator(new QIntValidator(0, 9999999, this));
    ui->y0->setValidator(new QIntValidator(0, 9999999, this));
    ui->SquareSize->setValidator(new QIntValidator(0, 9999999, this));
    ui->x0Rect->setValidator(new QIntValidator(0, 9999999, this));
    ui->y0Rect->setValidator(new QIntValidator(0, 9999999, this));
    ui->Lx->setValidator(new QIntValidator(0, 9999999, this));
    ui->Ly->setValidator(new QIntValidator(0, 9999999, this));
    ui->dz->setValidator(new QDoubleValidator(0.0, 9999999.0, 3, this));
    ui->dx->setValidator(new QDoubleValidator(0.0, 9999999.0, 3, this));
    ui->z0->setValidator(new QDoubleValidator(0.0, 9999999.0, 3, this));
    ui->lambda->setValidator(new QDoubleValidator(0.0, 9999999.0, 3, this));
    ui->Sparsity->setValidator(new QDoubleValidator(0.0, 9999999.0, 3, this));
    ui->TV->setValidator(new QDoubleValidator(0.0, 9999999.0, 3, this));
    ui->MinVoxelIntensity->setValidator(new QIntValidator(0, 9999999, this));
    ui->MinVoxelSz->setValidator(new QIntValidator(0, 9999999, this));
    ui->SegCloseSz->setValidator(new QIntValidator(0, 9999999, this));

    // Initial text
    ui->MinVoxelIntensity->setPlaceholderText("in pixels.");
    ui->MinVoxelSz->setPlaceholderText("in pixels.");
    ui->SegCloseSz->setPlaceholderText("in pixels.");
    ui->OutputDir->setPlaceholderText("Directory for the generated outputs.");
    ui->BGName->setPlaceholderText("In C-style formatting (e.g. background_%04.tif). Fills automatically when bg type is selected.");
    ui->HoloImgFileFormat->setPlaceholderText("In C-style formatting (e.g. data_%04.tif). Fills automatically when a sample input image is selected.");
    ui->HoloImgPath->setPlaceholderText("Select an image from the image folder.");
    ui->HoloBGpath->setPlaceholderText("Select the directory for containing background images(s).");
    ui->HoloStartImage->setPlaceholderText("Default fills automatically once Image Path is selected.");
    ui->HoloEndImg->setPlaceholderText("Default fills automatically once Image Path is selected.");
    ui->WindowSize->setPlaceholderText("# of frames for moving average.");
    ui->HololIncrement->setPlaceholderText("To skip frames at a regular interval");
    ui->z0->setPlaceholderText("um.");
    ui->lambda->setPlaceholderText("um.");
    ui->dx->setPlaceholderText("um.");
    ui->dz->setPlaceholderText("um.");


    // initialize using the BG tab information
    imgPathShared = ui->ImgPath->text(); // imgpath
    ui->HoloImgPath->setText(imgPathShared);
    imgFileFormatShared = ui->ImgFileFormat->text(); // img name format
    ui->HoloImgFileFormat->setText(imgFileFormatShared);
    imgStartIdXShared = ui->StartImage->text(); // start img idx
    ui->HoloStartImage->setText(imgStartIdXShared);
    imgEndIdXShared = ui->EndImage->text(); // start img idx
    ui->HoloEndImg->setText(imgEndIdXShared);
    bgPathShared = ui->BGpath->text(); // bg path
    ui->HoloBGpath->setText(bgPathShared);

    // default background names
    connect(ui->ReconNoBG, &QRadioButton::toggled, this, [this](bool checked){
        if (checked) {
            if (checked) ui->BGName->clear();
        }
    });

    connect(ui->ReconMeanBG, &QRadioButton::toggled, this, [this](bool checked){
        if (checked) {
            ui->BGName->setText("background.png");
        }
    });

    connect(ui->ReconMedianBG, &QRadioButton::toggled, this, [this](bool checked){
        if (checked) {
            ui->BGName->setText("background_%04d.png");
        }
    });

    connect(ui->ReconMovingBG, &QRadioButton::toggled, this, [this](bool checked){
        if (checked) {
            ui->BGName->setText("background_%04d.png");
        }
    });

    connect(ui->ReconCorrelationBG, &QRadioButton::toggled, this, [this](bool checked){
        if (checked) {
            ui->BGName->setText("background_%04d.png");
        }
    });

    // enable the cropping/sizing options
    connect(ui->CenterRadio, &QRadioButton::toggled, this, [this](bool checked){
        // Enable the CenterOptions group only when CenterRadio is checked
        ui->CenterOptions->setEnabled(checked);
    });
    connect(ui->RectRadio, &QRadioButton::toggled, this, [this](bool checked){
        // Enable the CenterOptions group only when CenterRadio is checked
        ui->RectangleOptions->setEnabled(checked);
    });
    // enable the segmentation options
    connect(ui->ExtractCentroids, &QCheckBox::toggled, this, [this](bool checked){
        ui->SegmentationOptions->setEnabled(checked);
    });

    // check everytime things are disabled/enabled at the startup
    ui->CenterOptions->setEnabled(ui->CenterRadio->isChecked());
    ui->RectangleOptions->setEnabled(ui->RectRadio->isChecked());
    ui->SegmentationOptions->setEnabled(ui->ExtractCentroids->isChecked());

    // Image path (from the sample image, we can extract the number of images, image prefix, etc. to populate the related fields)
    connect(ui->HoloImgPathBrowse, &QPushButton::clicked, this, [this]() {
        QString sampleFile = QFileDialog::getOpenFileName(
            this,
            tr("Select Sample Image"),
            ui->HoloImgPath->text(),
            tr("Images (*.tif *.jpg *.png *.bmp *.tiff)")
            );

        if (sampleFile.isEmpty())
            return;

        QFileInfo fileInfo(sampleFile);
        QString folderPath = fileInfo.absolutePath();
        QString fileName = fileInfo.fileName();   // e.g. "Im_000004.jpg"
        QString suffix = fileInfo.suffix();       // e.g. "jpg"

        // Ensure the folder path ends with a separator (RIHVR expects '/' or '\' in the very end of the path)
        // Always convert to forward slashes
        folderPath.replace('\\', '/');
        // Ensure it ends with a forward slash
        if (!folderPath.endsWith('/'))
            folderPath += '/';

        // Match the LAST numeric group in the filename
        QRegularExpression lastNumRx("(\\d+)(?!.*\\d)");
        QRegularExpressionMatch m = lastNumRx.match(fileName);
        if (!m.hasMatch()) {
            QMessageBox::warning(this, tr("Error"), tr("No numeric part found in file name."));
            return;
        }

        QString numberStr = m.captured(1);        // e.g. "000004"
        int numDigits = numberStr.length();       // e.g. 6
        int startIndex = m.capturedStart(1);      // index of that numeric group in fileName

        // prefix = part before the numeric section, suffix (extension) already known
        QString prefix = fileName.left(startIndex);                // e.g. "Im_"
        QString extension = suffix;                                // e.g. "jpg"

        // Build the printf-style pattern safely: e.g. "Im_%06d.jpg"
        QString filePattern = prefix + "%" + QString("0%1d").arg(numDigits) + "." + extension;

        // Now scan the directory for files with same prefix and extension to find min/max index.
        QDir dir(folderPath);
        // Build a name filter like "Im_*.jpg"
        QString nameFilter = prefix + "*" + "." + extension;
        QStringList nameFilters;
        nameFilters << nameFilter;
        dir.setNameFilters(nameFilters);

        QStringList fileList = dir.entryList(QDir::Files, QDir::Name);
        if (fileList.isEmpty()) {
            // still update UI with the single-file info
            ui->HoloImgPath->setText(folderPath);
            ui->HoloImgFileFormat->setText(filePattern);
            ui->HoloStartImage->setText(QString::number(numberStr.toInt()));
            ui->HoloEndImg->setText(QString::number(numberStr.toInt()));
            return;
        }

        int minNum = numberStr.toInt();
        int maxNum = numberStr.toInt();

        // Use regex to extract the last numeric group in each filename
        QRegularExpression numRx("(\\d+)(?!.*\\d)");
        for (const QString &fn : fileList) {
            QRegularExpressionMatch mm = numRx.match(fn);
            if (mm.hasMatch()) {
                int n = mm.captured(1).toInt();
                if (n < minNum) minNum = n;
                if (n > maxNum) maxNum = n;
            }
        }
        //

        // Update UI fields in the processing tab according to what we figured out from the sample image path
        ui->HoloImgPath->setText(folderPath);
        ui->HoloImgFileFormat->setText(filePattern);
        ui->HoloStartImage->setText(QString::number(minNum));
        ui->HoloEndImg->setText(QString::number(maxNum));
    });

    // Holo BG path -- also ensures that the path name ends with a slash
    connect(ui->HoloBGPathBrowse, &QPushButton::clicked, this, [this]() {
        QString dir = QFileDialog::getExistingDirectory(this, tr("Select background Folder"), ui->HoloBGpath->text());
        if (!dir.isEmpty()) {
            // Normalize path: forward slashes and trailing slash
            dir.replace('\\', '/');  // replace backslashes with forward slashes
            if (!dir.endsWith('/'))  // ensure trailing slash
                dir += '/';

            ui->HoloBGpath->setText(dir);
        }
    });

    // Holo output directory path -- also ensures that the path name ends with a slash
    connect(ui->OutputDirBrowse, &QPushButton::clicked, this, [this]() {
        QString dir = QFileDialog::getExistingDirectory(this, tr("Select Output Folder"), ui->OutputDir->text());
        if (!dir.isEmpty()) {
            // Normalize path: forward slashes and trailing slash
            dir.replace('\\', '/');  // replace backslashes with forward slashes
            if (!dir.endsWith('/'))  // ensure trailing slash
                dir += '/';

            ui->OutputDir->setText(dir);
        }
    });

    // When HoloImgPath changes, update ImgPath
    connect(ui->HoloImgPath, &QLineEdit::textChanged, this, [this](const QString &text) {
        static bool localSync = false;
        if (localSync) return; // prevent loop
        localSync = true;
        imgPathShared = text;
        ui->ImgPath->setText(text);
        localSync = false;
    });

    // When HoloImgFileFormat changes, update ImgFileFormat in the bg tab
    connect(ui->HoloImgFileFormat, &QLineEdit::textChanged, this, [this](const QString &text) {
        static bool localSync = false;
        if (localSync) return; // prevent loop
        localSync = true;
        imgFileFormatShared = text;
        ui->ImgFileFormat->setText(text);
        localSync = false;
    });

    // When HoloStartImage changes, update StartImage in the bg tab
    connect(ui->HoloStartImage, &QLineEdit::textChanged, this, [this](const QString &text) {
        static bool localSync = false;
        if (localSync) return; // prevent loop
        localSync = true;
        imgStartIdXShared = text;
        ui->StartImage->setText(text);
        localSync = false;
    });

    // When HoloEndImage changes, update EndImage in the bg tab
    connect(ui->HoloEndImg, &QLineEdit::textChanged, this, [this](const QString &text) {
        static bool localSync = false;
        if (localSync) return; // prevent loop
        localSync = true;
        imgEndIdXShared = text;
        ui->EndImage->setText(text);
        localSync = false;
    });

    // When HoloBGpath changes, update BGpath in the bg tab
    connect(ui->HoloBGpath, &QLineEdit::textChanged, this, [this](const QString &text) {
        static bool localSync = false;
        if (localSync) return; // prevent loop
        localSync = true;
        bgPathShared = text;
        ui->BGpath->setText(text);
        localSync = false;
    });

    //==== Process holograms button
    connect(ui->ProcessHolograms, &QPushButton::clicked, this, &RIHVR::computeHolographicReconstruction);

    // ================================================================== Tracking page =======================================================================//
    // ensure that the numeric fields only accept numbers
    ui->TrackStartIdX->setValidator(new QIntValidator(0, 9999999, this));
    ui->TrackEndIdX->setValidator(new QIntValidator(0, 9999999, this));
    ui->MinParticleSize->setValidator(new QIntValidator(0, 9999999, this));
    ui->MaxParticleSize->setValidator(new QIntValidator(0, 9999999, this));
    ui->MaxDisplacement->setValidator(new QIntValidator(0, 9999999, this));
    ui->MinTrajLength->setValidator(new QIntValidator(0, 9999999, this));
    ui->ParticleMemory->setValidator(new QIntValidator(0, 9999999, this));
    ui->Dimensions->setValidator(new QIntValidator(0, 9999999, this));
    // ui->ResizePercent->setValidator(new QDoubleValidator(0.0, 9999999.0, 3, this));
    // Initial text
    ui->CentroidDataPath->setPlaceholderText("Select a centroid data csv file from the centroids folder.");
    ui->CentroidDataFileFormat->setPlaceholderText("In C-style formatting (e.g. data_%04.tif). Fills automatically when a sample csv is selected.");
    ui->TrackStartIdX->setPlaceholderText("Start index for the centroid data. Default fills automatically once centroid data Path is selected.");
    ui->TrackEndIdX->setPlaceholderText("End index for the centroid data. Default fills automatically once centroid data Path is selected.");
    ui->MinParticleSize->setPlaceholderText("Minimum paritcle size (in voxels).");
    ui->MaxParticleSize->setPlaceholderText("Maximum paritcle size (in voxels).");
    ui->MaxDisplacement->setPlaceholderText("Maximum paritcle displacement (in voxels).");
    ui->MinTrajLength->setPlaceholderText("Length of the smallest trajectory (in number of frames).");
    ui->ParticleMemory->setPlaceholderText("Number of frames for which a disappeared particle should be remembered.");
    ui->Dimensions->setPlaceholderText("Dimensions of the data (2 or 3)");

    // When OutputDir changes in the processing tab, update CentroidDir to be OutputDir/Centroids/
    connect(ui->OutputDir, &QLineEdit::textChanged, this, [this](const QString &newOutDir) {
        QString centroidPath = QDir(newOutDir).filePath("Centroids");

        // Normalize path: always forward slashes and ensure trailing slash
        centroidPath.replace('\\', '/');  // enforce forward slashes
        if (!centroidPath.endsWith('/'))
            centroidPath += '/';

        ui->CentroidDataPath->setText(centroidPath);
    });

    // ensure that the centroidpath does exist
    QDir centroidPath(ui->CentroidDataPath->text());
    if (!centroidPath.exists())
        centroidPath.mkpath(".");

    // When OutputDir changes in the processing tab, update Tracks to be OutputDir/Tracks/
    connect(ui->OutputDir, &QLineEdit::textChanged, this, [this](const QString &newOutDir) {
        QString tracksPath = QDir(newOutDir).filePath("Tracks");

        // Normalize path: always forward slashes and ensure trailing slash
        tracksPath.replace('\\', '/');  // enforce forward slashes
        if (!tracksPath.endsWith('/'))
            tracksPath += '/';

        ui->TrackDirPath->setText(tracksPath);
    });

    // ensure that the centroidpath does exist
    QDir tracksPath(ui->TrackDirPath->text());
    if (!tracksPath.exists())
        tracksPath.mkpath(".");

    // Centroid path (from the sample centroid csv, we can extract the number of files, file name prefix, etc. to populate the related fields)
    connect(ui->CentroidDataBrowse, &QPushButton::clicked, this, [this]() {
        QString sampleFile = QFileDialog::getOpenFileName(
            this,
            tr("Select a sample centroids csv data file"),
            ui->HoloImgPath->text(),
            tr("Centroid data files (*.csv)")
            );

        if (sampleFile.isEmpty())
            return;

        QFileInfo fileInfo(sampleFile);
        QString folderPath = fileInfo.absolutePath();
        QString fileName = fileInfo.fileName();   // e.g. "Im_000004.jpg"
        QString suffix = fileInfo.suffix();       // e.g. "jpg"

        // Ensure the folder path ends with a separator (RIHVR expects '/' or '\' in the very end of the path)
        // Always convert to forward slashes
        folderPath.replace('\\', '/');
        // Ensure it ends with a forward slash
        if (!folderPath.endsWith('/'))
            folderPath += '/';

        // Match the LAST numeric group in the filename
        QRegularExpression lastNumRx("(\\d+)(?!.*\\d)");
        QRegularExpressionMatch m = lastNumRx.match(fileName);
        if (!m.hasMatch()) {
            QMessageBox::warning(this, tr("Error"), tr("No numeric part found in file name."));
            return;
        }

        QString numberStr = m.captured(1);        // e.g. "000004"
        int numDigits = numberStr.length();       // e.g. 6
        int startIndex = m.capturedStart(1);      // index of that numeric group in fileName

        // prefix = part before the numeric section, suffix (extension) already known
        QString prefix = fileName.left(startIndex);                // e.g. "Im_"
        QString extension = suffix;                                // e.g. "jpg"

        // Build the printf-style pattern safely: e.g. "Im_%06d.jpg"
        QString filePattern = prefix + "%" + QString("0%1d").arg(numDigits) + "." + extension;

        // Now scan the directory for files with same prefix and extension to find min/max index.
        QDir dir(folderPath);
        // Build a name filter like "Im_*.jpg"
        QString nameFilter = prefix + "*" + "." + extension;
        QStringList nameFilters;
        nameFilters << nameFilter;
        dir.setNameFilters(nameFilters);

        QStringList fileList = dir.entryList(QDir::Files, QDir::Name);
        if (fileList.isEmpty()) {
            // still update UI with the single-file info
            ui->CentroidDataPath->setText(folderPath);
            ui->CentroidDataFileFormat->setText(filePattern);
            ui->TrackStartIdX->setText(QString::number(numberStr.toInt()));
            ui->TrackEndIdX->setText(QString::number(numberStr.toInt()));
            return;
        }

        int minNum = numberStr.toInt();
        int maxNum = numberStr.toInt();

        // Use regex to extract the last numeric group in each filename
        QRegularExpression numRx("(\\d+)(?!.*\\d)");
        for (const QString &fn : fileList) {
            QRegularExpressionMatch mm = numRx.match(fn);
            if (mm.hasMatch()) {
                int n = mm.captured(1).toInt();
                if (n < minNum) minNum = n;
                if (n > maxNum) maxNum = n;
            }
        }
        //

        // Update UI fields in the processing tab according to what we figured out from the sample image path
        ui->CentroidDataPath->setText(folderPath);
        ui->CentroidDataFileFormat->setText(filePattern);
        ui->TrackStartIdX->setText(QString::number(minNum));
        ui->TrackEndIdX->setText(QString::number(maxNum));
    });

    //TrackDirPath path -- also ensures that the path name ends with a slash
    connect(ui->TrackDirBrowse, &QPushButton::clicked, this, [this]() {
        QString dir = QFileDialog::getExistingDirectory(this, tr("Select a folder to save the computed tracks"), ui->TrackDirPath->text());
        if (!dir.isEmpty()) {
            // Normalize path: forward slashes and trailing slash
            dir.replace('\\', '/');  // replace backslashes with forward slashes
            if (!dir.endsWith('/'))  // ensure trailing slash
                dir += '/';

            ui->TrackDirPath->setText(dir);
        }
    });

    // initialize the centroid start and end index using the holographic reconstruction start and end index
    connect(ui->HoloStartImage, &QLineEdit::textChanged, this, [this](const QString &newText) {
        ui->TrackStartIdX->setText(newText);
        trackStartIdXShared = newText;
    });
    connect(ui->HoloEndImg, &QLineEdit::textChanged, this, [this](const QString &newText) {
        ui->TrackEndIdX->setText(newText);
        trackEndIdXShared = newText;
    });

    //==== compute tracks button
    connect(ui->ComputeTracks, &QPushButton::clicked, this, &RIHVR::computeTracks);

    // ================================================================== Batch Automation page ==================================================================//
    ui->RIHVRFilesDir->setPlaceholderText("Patch to the directory containing .rihvr files to batch process.");

    //.rihvr files path -- also ensures that the path name ends with a slash
    connect(ui->RIHVRFilesBrowseButton, &QPushButton::clicked, this, [this]() {
        QString dir = QFileDialog::getExistingDirectory(this, tr("Select a folder containing .rihvr files to batch process."), ui->RIHVRFilesDir->text());
        if (!dir.isEmpty()) {
            // Normalize path: forward slashes and trailing slash
            dir.replace('\\', '/');  // replace backslashes with forward slashes
            if (!dir.endsWith('/'))  // ensure trailing slash
                dir += '/';

            ui->RIHVRFilesDir->setText(dir);
        }
    });

    // ==== pipeline automatic batching
    // When background computation finishes
    connect(this, &RIHVR::computeBGFinished, this, [this](bool success) {
        if (!m_pipelineRunning) return;

        if (!success) {
            ui->Console->append("<span style='color: red;'><b>Background computation failed. Pipeline stopped.</b></span>");
            ui->Console->append("<span style='color: white;'></span>");
            emit pipelineFinished(false);  // signal batch that this file failed
            return;
        }

        ui->Console->append("<span style='color: lime;'>Background completed successfully.</span>");
        ui->Console->append("<span style='color: white;'></span>");

        if (ui->ProcessHoloCheckAuto->isChecked()) {
            ui->Console->append("<span style='color: cyan;'>Auto: Starting Holo computation...</span>");
            ui->Console->append("<span style='color: white;'></span>");
            computeHolographicReconstruction();
        }
        else if (ui->ComputeTracksCheckAuto->isChecked()) {
            ui->Console->append("<span style='color: cyan;'>Skipping Holo, starting Track computation...</span>");
            ui->Console->append("<span style='color: white;'></span>");
            computeTracks();
        }
        else {
            // No further stages, pipeline finished
            emit pipelineFinished(true);
        }
    });

    // When Holo computation finishes
    connect(this, &RIHVR::computeHoloFinished, this, [this](bool success) {
        if (!m_pipelineRunning) return;
        if (!success) {
            ui->Console->append("<span style='color: red;'><b>Holo computation failed. Pipeline stopped.</b></span>");
            ui->Console->append("<span style='color: white;'></span>");
            emit pipelineFinished(false);
            return;
        }

        ui->Console->append("<span style='color: lime;'>Holo computation completed successfully.</span>");
        ui->Console->append("<span style='color: white;'></span>");

        if (ui->ComputeTracksCheckAuto->isChecked()) {
            ui->Console->append("<span style='color: cyan;'>Auto: Starting Track computation...</span>");
            ui->Console->append("<span style='color: white;'></span>");
            computeTracks();
        } else {
            // No Tracks stage, pipeline done
            emit pipelineFinished(true);
        }
    });


    // When Tracks computation finishes
    connect(this, &RIHVR::computeTracksFinished, this, [this](bool success) {
        if (!m_pipelineRunning) return;

        if (success) {
            ui->Console->append("<span style='color: lime;'><b>All computations completed successfully!</b></span>");
        } else {
            ui->Console->append("<span style='color: red;'><b>Track computation failed.</b></span>");
        }
        ui->Console->append("<span style='color: white;'></span>");

        // Pipeline done
        emit pipelineFinished(success);
    });


    // AutoProcessButton button
    connect(ui->AutoProcessButton, &QPushButton::clicked,this, &RIHVR::on_ProcessAll_clicked);

    // load the next rihvr file if the current .rihvr is done
    connect(this, &RIHVR::pipelineFinished, this, [this](bool success) {
        if (!m_batchRunning) return;  // ignore if not in batch

        if (!m_stopBatch) {
            m_currentBatchIndex++;
            processNextRIHVRFile();
        } else {
            ui->Console->append("<span style='color: red;'><b>Batch halted by user.</b></span>");
            ui->Console->append("<span style='color: white;'></span>");
            m_batchRunning = false;
        }
    });


    // .rihvr processing button
    connect(ui->ProcessAllRIHVRFiles, &QPushButton::clicked,this, &RIHVR::on_ProcessAllRIHVRFiles_clicked);

    // ================================================================== Visualzation page ==================================================================//
    // ensure that the numeric fields only accept numbers
    // Initial text
    ui->TracksVizPath->setPlaceholderText("Select the tracks csv file.");
    ui->HoloVizImgPath->setPlaceholderText("Path to the holograms.");
    ui->PhaseVizImgPath->setPlaceholderText("Path to saved projections.");

    // When HoloImgPath and stuff changes in processing, update HoloVizPath, and connect the slider to it
    connect(ui->HoloImgPath,        &QLineEdit::textChanged,  this, &RIHVR::updateHoloVizPath);
    connect(ui->HoloImgFileFormat,  &QLineEdit::textChanged,  this, &RIHVR::updateHoloVizPath);
    connect(ui->HoloStartImage,  &QLineEdit::textChanged,  this, &RIHVR::updateHoloVizPath);
    connect(ui->HoloEndImg,  &QLineEdit::textChanged,  this, &RIHVR::updateHoloVizPath);
    connect(ui->HoloSlider,         &QSlider::valueChanged,   this, &RIHVR::updateHoloVizPath);
    connect(ui->HololIncrement,      &QLineEdit::textChanged,  this, &RIHVR::updateHoloVizPath);
    // slide to the image selected as the start image automatically
    connect(ui->HoloStartImage, &QLineEdit::textChanged, this, [this](const QString &text) {
        bool ok = false;
        int start = text.toInt(&ok);
        if (ok)
            ui->HoloSlider->setValue(start);
    });

    // When HoloImgPath and stuff changes in processing, update PhaseVizPath, and connect the slider to it
    connect(ui->OutputDir,        &QLineEdit::textChanged,  this, &RIHVR::updatePhaseVizPath);
    connect(ui->HoloStartImage,   &QLineEdit::textChanged,  this, &RIHVR::updatePhaseVizPath);
    connect(ui->HoloEndImg,       &QLineEdit::textChanged,  this, &RIHVR::updatePhaseVizPath);
    connect(ui->HololIncrement,    &QLineEdit::textChanged,  this, &RIHVR::updatePhaseVizPath);
    connect(ui->PhaseSlider,      &QSlider::valueChanged,   this, &RIHVR::updatePhaseVizPath);
    connect(ui->XYProj,           &QRadioButton::toggled,   this, &RIHVR::updatePhaseVizPath);
    connect(ui->YZProj,           &QRadioButton::toggled,   this, &RIHVR::updatePhaseVizPath);
    connect(ui->XZProj,           &QRadioButton::toggled,   this, &RIHVR::updatePhaseVizPath);
    // slide to the image selected as the start image automatically
    connect(ui->HoloStartImage, &QLineEdit::textChanged, this, [this](const QString &text) {
        bool ok = false;
        int start = text.toInt(&ok);
        if (ok)
            ui->PhaseSlider->setValue(start);
    });


    // When OutputDir changes in the processing tab, update Phase to be OutputDir/Projections/
    connect(ui->OutputDir, &QLineEdit::textChanged, this, [this](const QString &newOutDir) {
        QString phasePath = QDir(newOutDir).filePath("Projections");

        // Normalize path: always forward slashes and ensure trailing slash
        phasePath.replace('\\', '/');  // enforce forward slashes
        if (!phasePath.endsWith('/'))
            phasePath += '/';

        ui->PhaseVizImgPath->setText(phasePath);
    });

    ui->PhaseImgViz->setScaledContents(false); // to stop scaling of the image to fit the Qlabel
    ui->HoloImgViz->setScaledContents(false); // to stop scaling of the image to fit the Qlabel


    // When trackspath changes in tracking, update tracksVizPath
    connect(ui->TrackDirPath, &QLineEdit::textChanged, this, [this](const QString &text) {
        static bool localSync = false;
        if (localSync) return; // prevent loop
        localSync = true;
        trackPathShared = text;
        // Append the CSV filename
        QString csvPath = QDir(text).filePath("tracked_particles.csv");
        ui->TracksVizPath->setText(csvPath);
        localSync = false;
    });

    // ensure that HoloVizImgPath updates right when we check sync
    connect(ui->SyncWithHolo, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked) {
            // Align PhaseSlider to HoloSlider
            int holoVal = ui->HoloSlider->value();
            QSignalBlocker b(ui->PhaseSlider); // prevent recursion
            ui->PhaseSlider->setValue(holoVal);
            // Update both visualization paths immediately
            updateHoloVizPath();
            updatePhaseVizPath();
        }
    });


    // connect the contrast control buttons
    connect(ui->ContrastLO, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &RIHVR::updateHoloImageDisplay);
    connect(ui->ContrastHI, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &RIHVR::updateHoloImageDisplay);


    // tracks browse button
    connect(ui->TracksVizBrowseButton, &QPushButton::clicked, this, [this]() {
        QString sampleFile = QFileDialog::getOpenFileName(
            this,
            tr("Select a tracks csv data file"),
            ui->TracksVizPath->text(),
            tr("Tracks (*.csv)")
            );
        if (!sampleFile.isEmpty()) {
            ui->TracksVizPath->setText(sampleFile);
        }
    });

    // look for the startup file
    loadStartupPath();

    // sync the holo and phase sliders for the visualization page
    setupSliderSync();

    connect(ui->HoloVizImgBrowseButton, &QPushButton::clicked, this, &RIHVR::on_HoloVizImgBrowseButton_clicked);

    // Ensure one projection radio is active and initialize PhaseViz path
    if (ui->XYProj->isChecked() || ui->YZProj->isChecked() || ui->XZProj->isChecked()) {
        updatePhaseVizPath();
    }

    // if HoloVizImgPath changes, load the appropriate image
    connect(ui->HoloVizImgPath, &QLineEdit::textChanged, this, &RIHVR::updateHoloImageDisplay);

    // if PhaseVizImgPath changes, load the appropriate image
    connect(ui->PhaseVizImgPath, &QLineEdit::textChanged, this, &RIHVR::updatePhaseImageDisplay);

    // defining the min and max of the concentration slider (as a percentage)
    ui->TrackConcentration->setRange(0, 100);
    ui->TrackConcentration->setValue(100); // start at full

    // defining the min and max of the time slider (as a percentage)
    ui->TimeSlider->setRange(0, 100);
    ui->TimeSlider->setValue(100); // start at full

    // plotting the tracks
    //clearing the layout
    if (ui->TracksViz->layout())
    {
        QLayout* oldLayout = ui->TracksViz->layout();
        QLayoutItem* item;
        while ((item = oldLayout->takeAt(0)) != nullptr)
        {
            delete item->widget();
            delete item;
        }
        delete oldLayout;
    }

    // Initialize OpenGL tracks widget inside the UI container
    initializeTracksPlot();

    // Connect text field to automatic loading
    connect(ui->TracksVizPath, &QLineEdit::editingFinished,
            this, &RIHVR::loadTracksFromTextField);

    // Connect PlotTracks button
    connect(ui->PlotTracksButton, &QPushButton::clicked,
            this, &RIHVR::loadTracksFromTextField);  // reuse same slot

    // Connect the concentration slider
    connect(ui->TrackConcentration, &QSlider::valueChanged,
            this, &RIHVR::on_TrackConcentration_valueChanged);

    // Connect the time slider
    connect(ui->TimeSlider, &QSlider::valueChanged,
            this, &RIHVR::on_TimeSlider_valueChanged);




}

//============================================================================================================================================================================//
// ======================================================================= Defining functions ================================================================================//
//============================================================================================================================================================================//

// ================================================================== Miscellaneous functions ==================================================================//
// closing the GUI
RIHVR::~RIHVR()
{
    delete ui;
}

// RIHVR startup file check
void RIHVR::loadStartupPath()
{
    QString startupPathFile = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)
    + "/RIHVR_FFIL/startuppath.txt";

    QFile f(startupPathFile);
    if (!f.exists()) {
        QMessageBox::warning(this,
                             tr("Startup Path Not Found"),
                             tr("Startup path file not found in:\n%1").arg(startupPathFile));
        return;
    }

    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::warning(this,
                             tr("Startup Path Error"),
                             tr("Failed to open startup path file:\n%1").arg(startupPathFile));
        return;
    }

    QTextStream in(&f);
    QString defaultRihvrPath = in.readLine().trimmed();
    f.close();

    if (defaultRihvrPath.isEmpty()) {
        QMessageBox::warning(this,
                             tr("Startup Path Empty"),
                             tr("Default RIHVR path in startuppath.txt is empty."));
        return;
    }

    // confirm which JSON is being loaded
    qDebug() << "Default RIHVR settings file:" << defaultRihvrPath;
    //ui->Console->append(QString("Default RIHVR settings file: %1").arg(defaultRihvrPath));
    ui->Console->append(QString("<span style='color: cyan;'>Default RIHVR settings file: %1</span>").arg(defaultRihvrPath));
    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages

    // load the .rihvr settings file
    loadStateFromFile(defaultRihvrPath);

    // store the path in main window variable
    currentDefaultRihvrPath = defaultRihvrPath;

    // update the SettingsDialog UI if the dialog exists
    if (settingsdialog) {
        settingsdialog->setDefaultJsonPath(defaultRihvrPath);
    }
}

// ================================================================== Functions for the Menu Bar ==================================================================//
// About dialog box
void RIHVR::showAboutDialog()
{
    QString text =
        "<p>Made with ❤️ by <b>Flow Field Imaging Laboratory</b>.</p>"
        "<p>Please visit: "
        "<a href='https://www.jiaronghonglab.com/'>www.jiaronghonglab.com</a></p>";

    QMessageBox aboutBox;
    aboutBox.setWindowTitle("About RIHVR");
    aboutBox.setTextFormat(Qt::RichText);  // allows HTML formatting
    aboutBox.setText(text);
    aboutBox.setStandardButtons(QMessageBox::Ok);
    aboutBox.setIcon(QMessageBox::Information);
    aboutBox.exec();
}

void RIHVR::showLicenseDialog()
{
    QString text =
        "<p><b>RIHVR</b></p>"
        "<p>Copyright (C) 2025-2026 Gauresh Raj Jassal and Jiarong Hong</p>"
        "<p>This program is free software: you can redistribute it and/or modify "
        "it under the terms of the <b>GNU General Public License</b> as published by "
        "the Free Software Foundation, either version 3 of the License, or "
        "(at your option) any later version.</p>"

        "<p>This program is distributed in the hope that it will be useful, "
        "but <b>WITHOUT ANY WARRANTY</b>; without even the implied warranty of "
        "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the "
        "GNU General Public License for more details.</p>"
        "<p>You should have received a copy of the GNU General Public License "
        "along with this program. If not, see "
        "<a href='https://www.gnu.org/licenses/gpl-3.0.html'>"
        "https://www.gnu.org/licenses/gpl-3.0.html</a>.</p>";
        /*
        "<p><b>MIT License</b></p>"
        "<p>Copyright 2026 Gauresh Raj Jassal and Jiarong Hong</p>"
        "<p>Permission is hereby granted, free of charge, to any person obtaining a copy "
        "of this software and associated documentation files (the “Software”), to deal "
        "in the Software without restriction, including without limitation the rights "
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies "
        "of the Software, and to permit persons to whom the Software is furnished to do so, "
        "subject to the following conditions:</p>"
        "<p>The above copyright notice and this permission notice shall be included in all "
        "copies or substantial portions of the Software.</p>"
        "<p>THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, "
        "INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR "
        "PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE "
        "FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, "
        "ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.</p>";
        */

    QMessageBox licenseBox;
    licenseBox.setWindowTitle("License");
    licenseBox.setTextFormat(Qt::RichText);  // allows HTML formatting
    licenseBox.setText(text);
    licenseBox.setStandardButtons(QMessageBox::Ok);
    licenseBox.setIcon(QMessageBox::Information);
    licenseBox.exec();
}


// save state function
void RIHVR::saveStateToFile()
{
    QSettings settings("FFIL", "RIHVR");  // org + app name
    QString lastDir = settings.value("lastDirectory", QDir::homePath()).toString();

    QString fileName = QFileDialog::getSaveFileName(
        this,
        "Save Parameter State",
        lastDir,
        "RIHVR State Files (*.rihvr);;All Files (*.*)"
        );

    if (fileName.isEmpty())
        return;

    settings.setValue("lastDirectory", QFileInfo(fileName).absolutePath());

    // --- Build JSON object ---
    QJsonObject state;

    // Add settings dialog paths
    state["DefaultJSON"] = settingsdialog->getDefaultJsonPath();
    state["ProcessExe"] = settingsdialog->getProcessExePath();
    state["SegmentationExe"] = settingsdialog->getSegmentationExePath();
    state["BackgroundExe"] = settingsdialog->getBackgroundExePath();
    state["CCBGExe"] = settingsdialog->getCCBGExePath();
    state["TrackingExe"] = settingsdialog->getTrackingExePath();
    state["OpenCVPath"] = settingsdialog->getOpenCVPath();
    state["CudaPath"] = settingsdialog->getCudaPath();

    // == Add Background tab settings
    state["ComputeBG"] = ui->YesBackgroundCompute->isChecked();
    if (ui->MeanBG->isChecked()) state["BGType"] = "Mean";
    else if (ui->MedianBG->isChecked()) state["BGType"] = "Median";
    else if (ui->MovingBG->isChecked()) state["BGType"] = "Moving";
    state["ImgPath"] = ui->ImgPath->text();
    state["BGPath"] = ui->BGpath->text();
    state["StartImage"] = ui->StartImage->text();
    state["EndImage"] = ui->EndImage->text();
    state["BGStart"] = ui->BGStart->text();
    state["BGEnd"] = ui->BGEnd->text();
    state["ImgFileFormat"] = ui->ImgFileFormat->text();
    state["WindowSize"] = ui->WindowSize->text();

    // == Add Processing tab settings
    state["HoloOutputPath"] = ui->OutputDir->text();
    state["HoloImgPath"] = ui->HoloImgPath->text();
    state["HoloBGPath"] = ui->HoloBGpath->text();
    state["HoloBGName"] = ui->BGName->text();
    state["HoloStartImage"] = ui->HoloStartImage->text();
    state["HoloEndImage"] = ui->HoloEndImg->text();
    state["HoloImgFileFormat"] = ui->HoloImgFileFormat->text();
    state["HoloImgIncrement"] = ui->HololIncrement->text();
    if (ui->ReconMeanBG->isChecked()) state["HoloBG"] = "Mean";
    else if (ui->ReconMedianBG->isChecked()) state["HoloBG"] = "Median";
    else if (ui->ReconMovingBG->isChecked()) state["HoloBG"] = "Moving";
    else if (ui->ReconCorrelationBG->isChecked()) state["HoloBG"] = "CorrelationBased";
    else if (ui->ReconNoBG->isChecked()) state["HoloBG"] = "None";
    state["z0_StartPlane_um"] = ui->z0->text();
    state["dz_StepSize_um"] = ui->dz->text();
    state["Nz_NumPlanes"] = ui->Nz->text();
    state["dx_PixelResolution_um"] = ui->dx->text();
    state["lambda_Wavelength_um"] = ui->lambda->text();
    state["SparsityWeight"] = ui->Sparsity->text();
    state["TVWeight"] = ui->TV->text();
    state["ZeroPadding"] = ui->ZeroPad->text();
    state["ZeroPadding"] = ui->ZeroPad->text();
    state["NumFISTAIter"] = ui->nFISTA->text();
    state["numTVIter"] = ui->nTV->text();
    if (ui->CenterRadio->isChecked()) state["Cropping_ResizingType"] = "Center";
    else if (ui->RectRadio->isChecked()) state["Cropping_ResizingType"] = "RectangleROI";
    else if (ui->NoCropping->isChecked()) state["Cropping_ResizingType"] = "None";
    if (ui->SubfoldersCheck->isChecked()) state["CreateSubFolders"] = "True"; else state["CreateSubFolders"] = "False";
    if (ui->PhaseProjCheck->isChecked()) state["SavePhaseProjections"] = "True"; else state["SavePhaseProjections"] = "False";
    if (ui->PhaseEachPlaneCheck->isChecked()) state["SaveEachPhasePlane"] = "True"; else state["SaveEachPhasePlane"] = "False";
    if (ui->EnhanceCheck->isChecked()) state["SaveEnhanced"] = "True"; else state["SaveEnhanced"] = "False";
    if (ui->VoxelDataCheck->isChecked()) state["SaveVoxelData"] = "True"; else state["SaveVoxelData"] = "False";
    if (ui->SaveFISTACheck->isChecked()) state["SaveVoxelData"] = "True"; else state["SaveVoxelData"] = "False";
    if (ui->ExtractCentroids->isChecked()) state["ExtractCentroids"] = "True"; else state["ExtractCentroids"] = "False";
    state["CenterX0"] = ui->x0->text();
    state["CenterY0"] = ui->y0->text();
    state["CenterL"] = ui->SquareSize->text();
    state["RectX0"] = ui->x0Rect->text();
    state["RectY0"] = ui->y0Rect->text();
    state["RectLx"] = ui->Lx->text();
    state["RectLy"] = ui->Ly->text();
    state["Segment_CloseSize"] = ui->SegCloseSz->text();
    state["Segment_MinimumIntensity"] = ui->MinVoxelIntensity->text();
    state["Segment_MinimumSize"] = ui->MinVoxelSz->text();
    state["CCBG_ResizeFactor"] = ui->ResizePercent->text();
    state["CCBG_MinFrames"] = ui->MinFrames->text();
    state["CCBG_MaxFrames"] = ui->MaxFrames->text();
    state["CCBG_SaveEnhanced"] = ui->SaveCCBGEnh->text();

    // == Add Tracking tab settings
    state["TracksOutputPath"] = ui->TrackDirPath->text();
    state["CentroidFileFormat"] = ui->CentroidDataFileFormat->text();
    state["CentroidDataPath"] = ui->CentroidDataPath->text();
    state["TracksStartIndex"] = ui->TrackStartIdX->text();
    state["TracksEndIndex"] = ui->TrackEndIdX->text();
    state["Tracking_MinParticleSize"] = ui->MinParticleSize->text();
    state["Tracking_MaxParticleSize"] = ui->MaxParticleSize->text();
    state["Tracking_MaxDisplacement"] = ui->MaxDisplacement->text();
    state["Tracking_MinTrajectoryLength"] = ui->MinTrajLength->text();
    state["Tracking_Memory"] = ui->ParticleMemory->text();
    state["Tracking_Dimensions"] = ui->Dimensions->text();
    if (ui->Print2Console->isChecked()) state["PrintTrackingOutput2Console"] = "True";

    // == Add Visualization tab settings
    state["HoloVizPath"] = ui->HoloVizImgPath->text();
    state["PhaseVizPath"] = ui->PhaseVizImgPath->text();
    state["TracksVizPath"] = ui->TracksVizPath->text();
    if (ui->HoloVizSubtractBG->isChecked()) state["SubtractBGHoloViz"] = "True";
    if (ui->SyncWithHolo->isChecked()) state["SyncHoloPhaseViz"] = "True";
    if (ui->XYProj->isChecked()) state["PhaseProjVizAxis"] = "XY";
    else if (ui->YZProj->isChecked()) state["PhaseProjVizAxis"] = "YZ";
    else if (ui->XZProj->isChecked()) state["PhaseProjVizAxis"] = "XZ";

    // == Add Batch tab settings
    state["BatchRIHVRDir"] = ui->RIHVRFilesDir->text();
    if (ui->ComputeBGCheckAuto->isChecked()) state["BatchBGCheck"] = "True";
    if (ui->ComputeTracksCheckAuto->isChecked()) state["BatchTracksCheck"] = "True";
    if (ui->ProcessHoloCheckAuto->isChecked()) state["BatchHoloCheck"] = "True";

    // Write JSON to file
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly)) {
        QMessageBox::warning(this, tr("Save State"), tr("Cannot open file for writing."));
        return;
    }

    QJsonDocument doc(state);
    file.write(doc.toJson());
    file.close();

    QMessageBox::information(this, tr("Save State"), tr("Parameters saved successfully!"));
}

// load state function
void RIHVR::loadStateFromFile(const QString &filePath)
{
    QString fileName = filePath;

    if (fileName.isEmpty()) {
        // fallback to manual dialog if no path is provided
        QSettings settings("FFIL", "RIHVR");
        QString lastDir = settings.value("lastDirectory", QDir::homePath()).toString();

        fileName = QFileDialog::getOpenFileName(
            this,
            "Load Parameter State",
            lastDir,
            "RIHVR State Files (*.rihvr);;All Files (*.*)"
            );

        if (fileName.isEmpty())
            return;

        settings.setValue("lastDirectory", QFileInfo(fileName).absolutePath());
    }

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, tr("Load State"), tr("Cannot open file for reading."));
        return;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        QMessageBox::warning(this, tr("Load State"),
                             tr("Failed to parse JSON:\n%1").arg(parseError.errorString()));
        return;
    }

    if (!doc.isObject()) {
        QMessageBox::warning(this, tr("Load State"), tr("Invalid file format."));
        return;
    }

    QJsonObject state = doc.object();

    // restore settings ---
    if (settingsdialog) {
        if (state.contains("DefaultJSON"))
            settingsdialog->setDefaultJsonPath(state["DefaultJSON"].toString());
        if (state.contains("ProcessExe"))
            settingsdialog->setProcessExePath(state["ProcessExe"].toString());
        if (state.contains("SegmentationExe"))
            settingsdialog->setSegmentationExePath(state["SegmentationExe"].toString());
        if (state.contains("BackgroundExe"))
            settingsdialog->setBackgroundExePath(state["BackgroundExe"].toString());
        if (state.contains("CCBGExe"))
            settingsdialog->setCCBGExePath(state["CCBGExe"].toString());
        if (state.contains("TrackingExe"))
            settingsdialog->setTrackingExePath(state["TrackingExe"].toString());
        if (state.contains("OpenCVPath"))
            settingsdialog->setOpenCVPath(state["OpenCVPath"].toString());
        if (state.contains("CudaPath"))
            settingsdialog->setCudaPath(state["CudaPath"].toString());

        // (updating the registry) save to QSettings immediately ---
        QSettings settings("FFIL", "RIHVR");
        settings.setValue("defaultJsonPath", settingsdialog->getDefaultJsonPath());
        settings.setValue("processExePath", settingsdialog->getProcessExePath());
        settings.setValue("segmentationExePath", settingsdialog->getSegmentationExePath());
        settings.setValue("backgroundExePath", settingsdialog->getBackgroundExePath());
        settings.setValue("trackingExePath", settingsdialog->getTrackingExePath());
        settings.setValue("ccBGExePath", settingsdialog->getCCBGExePath());
        settings.setValue("OpenCVPath", settingsdialog->getOpenCVPath());
        settings.setValue("CudaPath", settingsdialog->getCudaPath());
    }

    // --- Restore BG page ---
    if (state.contains("ComputeBG"))
        ui->YesBackgroundCompute->setChecked(state["ComputeBG"].toBool());
    if (state.contains("CCBG_SaveEnhanced")) ui->SaveCCBGEnh->setChecked(state["CCBG_SaveEnhanced"].toBool());
    if (state.contains("BGType")) {
        QString bgType = state["BGType"].toString();
        if (bgType == "Mean") ui->MeanBG->setChecked(true);
        else if (bgType == "Median") ui->MedianBG->setChecked(true);
        else if (bgType == "Moving") ui->MovingBG->setChecked(true);
        else if (bgType == "Correlation") ui->CCBG->setChecked(true);
    }
    if (state.contains("ImgPath")) ui->ImgPath->setText(state["ImgPath"].toString());
    if (state.contains("BGPath")) ui->BGpath->setText(state["BGPath"].toString());
    if (state.contains("StartImage")) ui->StartImage->setText(state["StartImage"].toString());
    if (state.contains("EndImage")) ui->EndImage->setText(state["EndImage"].toString());
    if (state.contains("BGStart")) ui->BGStart->setText(state["BGStart"].toString());
    if (state.contains("BGEnd")) ui->BGEnd->setText(state["BGEnd"].toString());
    if (state.contains("ImgFileFormat")) ui->ImgFileFormat->setText(state["ImgFileFormat"].toString());
    if (state.contains("WindowSize")) ui->WindowSize->setText(state["WindowSize"].toString());
    if (state.contains("CCBG_ResizeFactor")) ui->ResizePercent->setText(state["CCBG_ResizeFactor"].toString());
    if (state.contains("CCBG_MinFrames")) ui->MinFrames->setText(state["CCBG_MinFrames"].toString());
    if (state.contains("CCBG_MaxFrames")) ui->MaxFrames->setText(state["CCBG_MaxFrames"].toString());

    // --- Restore Processing page ---
    if (state.contains("HoloOutputPath"))ui->OutputDir->setText(state["HoloOutputPath"].toString());
    if (state.contains("HoloImgPath"))ui->HoloImgPath->setText(state["HoloImgPath"].toString());
    if (state.contains("HoloBGPath"))ui->HoloBGpath->setText(state["HoloBGPath"].toString());
    if (state.contains("HoloBGName"))ui->BGName->setText(state["HoloBGName"].toString());
    if (state.contains("HoloStartImage"))ui->HoloStartImage->setText(state["HoloStartImage"].toString());
    if (state.contains("HoloEndImage"))ui->HoloEndImg->setText(state["HoloEndImage"].toString());
    if (state.contains("HoloImgFileFormat"))ui->HoloImgFileFormat->setText(state["HoloImgFileFormat"].toString());
    if (state.contains("HoloImgIncrement"))ui->HololIncrement->setText(state["HoloImgIncrement"].toString());
    // Restore HoloBG radio buttons
    if (state.contains("HoloBG")) {
        QString bgType = state["HoloBG"].toString();
        if (bgType == "Mean") ui->ReconMeanBG->setChecked(true);
        else if (bgType == "Median") ui->ReconMedianBG->setChecked(true);
        else if (bgType == "Moving") ui->ReconMovingBG->setChecked(true);
        else if (bgType == "None") ui->ReconNoBG->setChecked(true);
    }
    // Restore numeric / text fields
    if (state.contains("z0_StartPlane_um")) ui->z0->setText(state["z0_StartPlane_um"].toString());
    if (state.contains("dz_StepSize_um")) ui->dz->setText(state["dz_StepSize_um"].toString());
    if (state.contains("Nz_NumPlanes")) ui->Nz->setText(state["Nz_NumPlanes"].toString());
    if (state.contains("dx_PixelResolution_um")) ui->dx->setText(state["dx_PixelResolution_um"].toString());
    if (state.contains("lambda_Wavelength_um")) ui->lambda->setText(state["lambda_Wavelength_um"].toString());
    if (state.contains("SparsityWeight")) ui->Sparsity->setText(state["SparsityWeight"].toString());
    if (state.contains("TVWeight")) ui->TV->setText(state["TVWeight"].toString());
    if (state.contains("ZeroPadding")) ui->ZeroPad->setText(state["ZeroPadding"].toString());
    if (state.contains("NumFISTAIter")) ui->nFISTA->setText(state["NumFISTAIter"].toString());
    if (state.contains("numTVIter")) ui->nTV->setText(state["numTVIter"].toString());
    if (state.contains("Segment_CloseSize")) ui->SegCloseSz->setText(state["Segment_CloseSize"].toString());
    if (state.contains("Segment_MinimumIntensity")) ui->MinVoxelIntensity->setText(state["Segment_MinimumIntensity"].toString());
    if (state.contains("Segment_MinimumSize")) ui->MinVoxelSz->setText(state["Segment_MinimumSize"].toString());
    // Restore cropping type radio buttons
    if (state.contains("Cropping_ResizingType")) {
        QString cropType = state["Cropping_ResizingType"].toString();
        if (cropType == "Center") ui->CenterRadio->setChecked(true);
        else if (cropType == "RectangleROI") ui->RectRadio->setChecked(true);
        else if (cropType == "None") ui->NoCropping->setChecked(true);
    }
    // Restore checkboxes
    if (state.contains("CreateSubFolders")) ui->SubfoldersCheck->setChecked(state["CreateSubFolders"].toString() == "True");
    if (state.contains("SavePhaseProjections")) ui->PhaseProjCheck->setChecked(state["SavePhaseProjections"].toString() == "True");
    if (state.contains("SaveEachPhasePlane")) ui->PhaseEachPlaneCheck->setChecked(state["SaveEachPhasePlane"].toString() == "True");
    if (state.contains("SaveEnhanced")) ui->EnhanceCheck->setChecked(state["SaveEnhanced"].toString() == "True");
    if (state.contains("SaveVoxelData")) ui->VoxelDataCheck->setChecked(state["SaveVoxelData"].toString() == "True");
    if (state.contains("ExtractCentroids")) ui->ExtractCentroids->setChecked(state["ExtractCentroids"].toString() == "True");
    // Restore rectangle / center coordinates
    if (state.contains("CenterX0")) ui->x0->setText(state["CenterX0"].toString());
    if (state.contains("CenterY0")) ui->y0->setText(state["CenterY0"].toString());
    if (state.contains("CenterL")) ui->SquareSize->setText(state["CenterL"].toString());
    if (state.contains("RectX0")) ui->x0Rect->setText(state["RectX0"].toString());
    if (state.contains("RectY0")) ui->y0Rect->setText(state["RectY0"].toString());
    if (state.contains("RectLx")) ui->Lx->setText(state["RectLx"].toString());
    if (state.contains("RectLy")) ui->Ly->setText(state["RectLy"].toString());


    // --- Restore Tracking page ---
    if (state.contains("PrintTrackingOutput2Console")) ui->Print2Console->setChecked(state["PrintTrackingOutput2Console"].toString() == "True");
    if (state.contains("TracksOutputPath"))ui->TrackDirPath->setText(state["TracksOutputPath"].toString());
    if (state.contains("CentroidFileFormat"))ui->CentroidDataFileFormat->setText(state["CentroidFileFormat"].toString());
    if (state.contains("CentroidDataPath"))ui->CentroidDataPath->setText(state["CentroidDataPath"].toString());
    if (state.contains("TracksStartIndex"))ui->TrackStartIdX->setText(state["TracksStartIndex"].toString());
    if (state.contains("TracksEndIndex"))ui->TrackEndIdX->setText(state["TracksEndIndex"].toString());
    if (state.contains("Tracking_MinParticleSize"))ui->MinParticleSize->setText(state["Tracking_MinParticleSize"].toString());
    if (state.contains("Tracking_MaxParticleSize"))ui->MaxParticleSize->setText(state["Tracking_MaxParticleSize"].toString());
    if (state.contains("Tracking_MaxDisplacement"))ui->MaxDisplacement->setText(state["Tracking_MaxDisplacement"].toString());
    if (state.contains("Tracking_MinTrajectoryLength"))ui->MinTrajLength->setText(state["Tracking_MinTrajectoryLength"].toString());
    if (state.contains("Tracking_Memory"))ui->ParticleMemory->setText(state["Tracking_Memory"].toString());
    if (state.contains("Tracking_Dimensions"))ui->Dimensions->setText(state["Tracking_Dimensions"].toString());

    // --- Restore Visualization page ---
    if (state.contains("SubtractBGHoloViz")) ui->HoloVizSubtractBG->setChecked(state["SubtractBGHoloViz"].toString() == "True");
    if (state.contains("SyncHoloPhaseViz")) ui->SyncWithHolo->setChecked(state["SyncHoloPhaseViz"].toString() == "True");
    if (state.contains("HoloVizPath"))ui->HoloVizImgPath->setText(state["HoloVizPath"].toString());
    if (state.contains("PhaseVizPath"))ui->PhaseVizImgPath->setText(state["PhaseVizPath"].toString());
    if (state.contains("TracksVizPath"))ui->TracksVizPath->setText(state["TracksVizPath"].toString());
    if (state.contains("PhaseProjVizAxis")) {
        QString projType = state["PhaseProjVizAxis"].toString();
        if (projType == "XY") ui->XYProj->setChecked(true);
        else if (projType == "YZ") ui->YZProj->setChecked(true);
        else if (projType == "XZ") ui->XZProj->setChecked(true);
    }

    // --- Restore Batch Automation page ---
    if (state.contains("BatchRIHVRDir"))ui->RIHVRFilesDir->setText(state["BatchRIHVRDir"].toString());
    if (state.contains("BatchBGCheck")) ui->ComputeBGCheckAuto->setChecked(state["BatchBGCheck"].toString() == "True");
    if (state.contains("BatchTracksCheck")) ui->ComputeTracksCheckAuto->setChecked(state["BatchTracksCheck"].toString() == "True");
    if (state.contains("BatchHoloCheck")) ui->ProcessHoloCheckAuto->setChecked(state["BatchHoloCheck"].toString() == "True");

    // show message only if loading manually
    if (filePath.isEmpty())
        QMessageBox::information(this, tr("Load State"), tr("Parameters loaded successfully!"));

    qDebug() << "Loaded .rihvr file:" << fileName;
    //ui->Console->append(QString("Loaded .rihvr file: %1").arg(fileName));
    ui->Console->append(QString("<span style='color: cyan;'>Loaded .rihvr file: %1</span>").arg(fileName));
    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
    ui->Console->append("<br>");

}

// ================================================================== Functions for the background page ==================================================================//
// disabling and enabling groups based on if the backgroud computation is on/off
void RIHVR::updateBackgroundTab()
{
    bool computeBG = ui->YesBackgroundCompute->isChecked();

    // Enable/disable the Background Type group box
    ui->BGTypebox->setEnabled(computeBG);

    // Enable/disable the Input/Output paths group box
    ui->BGComputeOptions->setEnabled(computeBG);

    // Enable moving background options only if BG type = Moving
    bool movingSelected = ui->MovingBG->isChecked();
    ui->MovingBGOptions->setEnabled(computeBG && movingSelected);

    // Enable CCBG options only if BG type = Correlation-based
    bool ccBGSelected = ui->CCBG->isChecked();
    ui->CCBGOptions->setEnabled(computeBG && ccBGSelected);

    // Enable/disable the compute background button based on all conditions
    ui->ComputeBGButton->setEnabled(canEnableComputeBGButton());
}

// conditions check for enabling the ComputeBGButton
bool RIHVR::canEnableComputeBGButton()
{
    // check if the compute background selected is yes
    if (!ui->YesBackgroundCompute->isChecked())
        return false;

    // check if the background type is selected
    bool bgTypeSelected = ui->MeanBG->isChecked() ||
                          ui->MedianBG->isChecked() ||
                          ui->CCBG->isChecked() ||
                          ui->MovingBG->isChecked();
    if (!bgTypeSelected)
        return false;

    // If Moving BG is selected, check if the moving BG options are filled out
    if (ui->MovingBG->isChecked())
    {
        if (ui->WindowSize->text().isEmpty())  // replace with your actual moving option widgets
            return false;
    }

    // check all the required file paths and options are filled in
    QList<QLineEdit*> requiredFields = {
        ui->ImgPath,
        ui->BGpath,
        ui->StartImage,
        ui->EndImage,
        ui->BGStart,
        ui->BGEnd,
        ui->ImgFileFormat
    };

    for (QLineEdit* le : requiredFields)
    {
        if (le->text().trimmed().isEmpty())
            return false;
    }

    return true;  // all conditions satisfied
}

// this function is for the mean and the median background computation
// we need to be a bit more clever for the moving one, as it needs to loop over multiple images
void RIHVR::computeBackground()
{
    // reset the stop button flag
    m_stopRequested = false; // reset at start

    // switch to console
    ui->tabWidget->setCurrentIndex(4);

    // Clear the console at the start
    //ui->Console->clear();
    //ui->Console->append("Beginning background calculation...");

    // Check BG Type selected
    bool isMeanBG   = ui->MeanBG->isChecked();
    bool isMedianBG = ui->MedianBG->isChecked();

    if (isMedianBG){
        ui->Console->append("<span style='color: cyan;'><b>Computing median background images...</b></span>");
        ui->Console->append(QString("<span style='color: cyan;'>Note: Median BG computation takes median of 5 frames.</span>"));
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
    }else{
        ui->Console->append("<span style='color: cyan;'><b>Computing mean background image...</b></span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
    }

    // Collect GUI values
    int numImages    = ui->EndImage->text().toInt();
    int startImage   = ui->StartImage->text().toInt();
    int bgStartImage = ui->BGStart->text().toInt();
    int bgEndImage   = ui->BGEnd->text().toInt();
    QString inputPath       = ui->ImgPath->text();
    QString inputFileFormat = ui->ImgFileFormat->text();
    QString outputPath      = ui->BGpath->text();
    bool useMedian          = isMedianBG;

    // Normalize paths
    if (!inputPath.endsWith('/'))  inputPath.replace('\\','/') += '/';
    if (!outputPath.endsWith('/')) outputPath.replace('\\','/') += '/';

    QString outputDir = ui->BGpath->text();
    QDir dir(outputDir);
    // ensures that the ouput directory exists
    if (!dir.exists()) {
        if (!dir.mkpath(".")) {  // "." means create the directory itself
            qWarning() << "Failed to create directory:" << outputPath;
        }
    }
    // create a directory to store the params file
    if (!dir.exists("params")) {dir.mkdir("params");}
    // Use the "params" subdirectory for parameter files
    QDir paramDir(dir.filePath("params"));
    QString paramFilePath = paramDir.filePath(QString("bg_%1.param").arg(1, 4, 10, QChar('0')));

    if (isMeanBG){QString paramFilePath = paramDir.filePath(QString("bg.params"));}

    // Generate parameter file
    if (!BGCompute::generateParameterFile(paramFilePath,
                                          numImages,
                                          startImage,
                                          bgStartImage,
                                          bgEndImage,
                                          inputPath,
                                          inputFileFormat,
                                          outputPath,
                                          useMedian))
    {
        QMessageBox::critical(this, "Error", "Failed to create parameter file.");
        return;
    }

    // Check settings dialog
    if (!settingsdialog) {
        QMessageBox::warning(this, "Error", "Settings dialog not available.");
        return;
    }

    QString bgExePath = settingsdialog->getBackgroundExePath();
    if (bgExePath.isEmpty()) {
        QMessageBox::warning(this, "Error", "make-background.exe path is not set. Please configure it in Settings.");
        emit computeBGFinished(false); // emit a signal that the process has failed
        return;
    }

    // Get CUDA/OpenCV paths
    QString cudaPath   = settingsdialog->getCudaPath();
    QString openCVPath = settingsdialog->getOpenCVPath();

    // Create process
    QProcess* process = new QProcess(this);

    // Set environment
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    if (!cudaPath.isEmpty())   env.insert("PATH", cudaPath + ";" + env.value("PATH"));
    if (!openCVPath.isEmpty()) env.insert("PATH", openCVPath + ";" + env.value("PATH"));
    process->setProcessEnvironment(env);

    // Connect signals for real-time output
    connect(process, &QProcess::readyReadStandardOutput, this, [this, process]() {
        QString output = process->readAllStandardOutput();
        ui->Console->moveCursor(QTextCursor::End);
        ui->Console->insertPlainText(output);
        ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());

    });

    connect(process, &QProcess::readyReadStandardError, this, [this, process]() {
        QString error = process->readAllStandardError();
        ui->Console->moveCursor(QTextCursor::End);
        ui->Console->insertPlainText(error);
        ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());
    });

    // Handle process finished
    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this, outputPath, isMeanBG](int exitCode, QProcess::ExitStatus exitStatus){
                bool success = false;
                if (isMeanBG) {
                    QString bgFile = outputPath + "background.png";
                    success = QFile::exists(bgFile);
                } else {
                    success = (exitStatus == QProcess::NormalExit && exitCode == 0);
                }

                if(success) {
                    ui->Console->append("<span style='color: green;'><b>Background computation completed successfully.</b></span>");
                    //QMessageBox::information(this, "Background Computation", "Completed successfully.");
                    // Reset color/format to default for future messages
                    ui->Console->append("<span style='color: white;'></span>");

                    emit computeBGFinished(true); // emit a signal that the process has finised successfully.

                } else {
                    ui->Console->append("<span style='color: red;'><b>Background computation failed.</b></span>");
                    QMessageBox::critical(this, "Background Computation",
                                          "Failed to compute background.\nCheck output folder or console output.");
                    // Reset color/format to default for future messages
                    ui->Console->append("<span style='color: white;'></span>");

                    emit computeBGFinished(false); // emit a signal that the process has failed
                }
            });


    // keep track of active processes, so that we can kill them using the stop button
    m_activeProcesses.append(process);

    // Remove from list when done
    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this, process](int, QProcess::ExitStatus){
                m_activeProcesses.removeAll(process);
                process->deleteLater();
            });

    // Start the process
    process->start(bgExePath, QStringList() << "-F" << paramFilePath);
    if (!process->waitForStarted()) {
        ui->Console->append("<span style='color: red;'><b>Failed to start make-background.exe.</b></span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
        QMessageBox::critical(this, "Error", "Failed to start make-background.exe");

        emit computeBGFinished(false); // emit a signal that the process has failed
        return;
    }
}

// this function is for the correlation-based background computation
void RIHVR::computeCCBackground()
{
    // reset the stop button flag
    m_stopRequested = false; // reset at start

    // switch to console
    ui->tabWidget->setCurrentIndex(4);

    // Clear the console at the start
    //ui->Console->clear();
    ui->Console->append("<span style='color: cyan;'><b>Computing correlation-based background images...</b></span>");
    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages

    // Collect GUI values
    int bgStartImage = ui->BGStart->text().toInt();
    int bgEndImage   = ui->BGEnd->text().toInt();
    double resizePct = ui->ResizePercent->text().toDouble();
    int minFrames   = ui->MinFrames->text().toInt();
    int maxFrames   = ui->MaxFrames->text().toInt();
    QString inputPath       = ui->ImgPath->text();
    QString inputFileFormat = ui->ImgFileFormat->text();
    QString outputPath      = ui->BGpath->text();
    bool saveEnhanced = ui->SaveCCBGEnh->isChecked();

    // Normalize paths
    if (!inputPath.endsWith('/'))  inputPath.replace('\\','/') += '/';
    if (!outputPath.endsWith('/')) outputPath.replace('\\','/') += '/';

    QString outputDir = ui->BGpath->text();
    QDir dir(outputDir);
    // ensures that the ouput directory exists
    if (!dir.exists()) {
        if (!dir.mkpath(".")) {  // "." means create the directory itself
            qWarning() << "Failed to create directory:" << outputPath;
        }
    }

    // Create a "params" subdirectory
    if (!dir.exists("params")) {dir.mkdir("params");}
    // Use the "params" subdirectory for parameter files
    QDir paramDir(dir.filePath("params"));
    QString paramFilePath = paramDir.filePath(QString("bg.params"));

    qDebug() << "Parms:" << bgStartImage << paramFilePath;
    // Generate parameter file
    if (!BGCompute::generateCCBGParameterFile(paramFilePath,
                                          bgStartImage,bgEndImage, resizePct, minFrames, maxFrames,
                                          inputPath,
                                          inputFileFormat,
                                          outputPath,
                                          saveEnhanced))
    {
        QMessageBox::critical(this, "Error", "Failed to create parameter file.");
        return;
    }

    // Check settings dialog
    if (!settingsdialog) {
        QMessageBox::warning(this, "Error", "Settings dialog not available.");
        return;
    }

    QString ccBGExePath = settingsdialog->getCCBGExePath();
    if (ccBGExePath.isEmpty()) {
        QMessageBox::warning(this, "Error", "ccbg.exe path is not set. Please configure it in Settings.");
        emit computeBGFinished(false); // emit a signal that the process has failed
        return;
    }

    // Get CUDA/OpenCV paths
    QString cudaPath   = settingsdialog->getCudaPath();
    QString openCVPath = settingsdialog->getOpenCVPath();

    // Create process
    QProcess* process = new QProcess(this);

    // Set environment
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    #ifdef Q_OS_WIN
        if (!cudaPath.isEmpty())   env.insert("PATH", cudaPath + ";" + env.value("PATH"));
        if (!openCVPath.isEmpty()) env.insert("PATH", openCVPath + ";" + env.value("PATH"));
    #else
        if (!cudaPath.isEmpty())   env.insert("LD_LIBRARY_PATH", cudaPath + ":" + env.value("LD_LIBRARY_PATH"));
        if (!openCVPath.isEmpty()) env.insert("LD_LIBRARY_PATH", openCVPath + ":" + env.value("LD_LIBRARY_PATH"));
    #endif
    process->setProcessEnvironment(env);

    // Connect signals for real-time output
    connect(process, &QProcess::readyReadStandardOutput, this, [this, process]() {
        QString output = process->readAllStandardOutput();
        ui->Console->moveCursor(QTextCursor::End);
        ui->Console->insertPlainText(output);
        ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());

    });

    connect(process, &QProcess::readyReadStandardError, this, [this, process]() {
        QString error = process->readAllStandardError();
        ui->Console->moveCursor(QTextCursor::End);
        ui->Console->insertPlainText(error);
        ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());
    });

    // Handle process finished
    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this, outputPath](int exitCode, QProcess::ExitStatus exitStatus) {
                bool success = false;

                if (exitStatus == QProcess::NormalExit && exitCode == 0)
                    success = true;

                if (success) {
                    ui->Console->append("<span style='color: green;'><b>Background computation completed successfully.</b></span>");
                    QMessageBox::information(this, "Background Computation", "Completed successfully.");

                    emit computeBGFinished(success); // emit a signal that the process has finised successfully.

                } else {
                    QString errMsg = QString("Background computation failed (exit code %1).").arg(exitCode);
                    ui->Console->append("<span style='color: red;'><b>" + errMsg + "</b></span>");
                    QMessageBox::critical(this, "Background Computation",
                                          "Failed to compute background.\nCheck output folder or console output.");
                    emit computeBGFinished(false); // emit a signal that the process has failed
                }

                ui->Console->append("<span style='color: white;'></span>");
            });



    // keep track of active processes, so that we can kill them using the stop button
    m_activeProcesses.append(process);

    // Remove from list when done
    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this, process](int, QProcess::ExitStatus){
                m_activeProcesses.removeAll(process);
                process->deleteLater();
            });

    // Start the process
    //qDebug() << "Launching:" << ccBGExePath << paramFilePath;
    process->start(ccBGExePath, QStringList() << paramFilePath);
    if (!process->waitForStarted()) {
        ui->Console->append("<span style='color: red;'><b>Failed to start ccbg.exe.</b></span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
        QMessageBox::critical(this, "Error", "Failed to start ccbg.exe");
        emit computeBGFinished(false); // emit a signal that the process has failed
        return;
    }
}

//========= Moving background function
void RIHVR::computeMovingBackground()
{
    // reset the stop button flag
    m_stopRequested = false; // reset at start

    // Switch to console tab
    ui->tabWidget->setCurrentIndex(4);

    // Collect GUI values
    int bgWindowSize  = ui->WindowSize->text().toInt();
    int bgFirstImage  = ui->BGStart->text().toInt();
    int bgEndImage    = ui->BGEnd->text().toInt();
    int numImages     = ui->EndImage->text().toInt();
    int startImage    = ui->StartImage->text().toInt();

    QString inputPath       = ui->ImgPath->text();
    QString inputFileFormat = ui->ImgFileFormat->text();
    QString outputPath      = ui->BGpath->text();
    // ensure that the output paths exists
    QString outputDir = ui->BGpath->text();
    QDir dir(outputDir);
    // ensures that the ouput directory exists
    if (!dir.exists()) {
        if (!dir.mkpath(".")) {  // "." means create the directory itself
            qWarning() << "Failed to create directory:" << outputPath;
        }
    }

    // Normalize paths
    if (!inputPath.endsWith('/'))  inputPath.replace('\\','/') += '/';
    if (!outputPath.endsWith('/')) outputPath.replace('\\','/') += '/';

    // Validation
    if (bgWindowSize <= 0) {
        ui->Console->append("<span style='color: red;'><b>Invalid background window size.</b></span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
        emit computeBGFinished(false); // emit a signal that the process has failed
        return;
    }
    if (bgFirstImage > bgEndImage) {
        ui->Console->append("<span style='color: red;'><b>Invalid frame range.</b></span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
        emit computeBGFinished(false); // emit a signal that the process has failed
        return;
    }

    int totalImages = bgEndImage - bgFirstImage + 1;
    if (bgWindowSize > totalImages)
        ui->Console->append("<span style='color: orange;'>Warning: window size larger than total count.</span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages

    //  Check settings dialog
    if (!settingsdialog) {
        QMessageBox::warning(this, "Error", "Settings dialog not available.");
        emit computeBGFinished(false); // emit a signal that the process has failed
        return;
    }

    QString bgExePath = settingsdialog->getBackgroundExePath();
    if (bgExePath.isEmpty()) {
        QMessageBox::warning(this, "Error", "make-background.exe path not set. Please configure it in Settings.");
        emit computeBGFinished(false); // emit a signal that the process has failed
        return;
    }

    QString cudaPath   = settingsdialog->getCudaPath();
    QString openCVPath = settingsdialog->getOpenCVPath();

    // === Create frame queue ===
    m_movingBGFrames.clear();
    for (int n = bgFirstImage; n <= bgEndImage; ++n)
        m_movingBGFrames.append(n);

    // Start processing first frame directly
    processNextMovingBG(bgWindowSize, bgFirstImage, bgEndImage,
                        numImages, startImage,
                        inputPath, inputFileFormat, outputPath,
                        bgExePath, cudaPath, openCVPath);
}
//========= subroutine the moving bg computation function
void RIHVR::processNextMovingBG(int bgWindowSize, int bgFirstImage, int bgEndImage,
                                int numImages, int startImage,
                                const QString& inputPath, const QString& inputFileFormat,
                                const QString& outputPath,
                                const QString& bgExePath, const QString& cudaPath,
                                const QString& openCVPath)
{
    if (m_movingBGFrames.isEmpty()) {
        ui->Console->append("<span style='color: green;'><b>Moving background computation complete.</b></span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
        ui->Console->append("<br>");

        emit computeBGFinished(true); // emit a signal that the process has finised successfully.

        return;
    }

    int n = m_movingBGFrames.takeFirst();

    // Compute start/end
    int lastValidStart = bgEndImage - bgWindowSize + 1;
    int half = bgWindowSize / 2;
    int proposedStart = n - half;
    int start = std::max(proposedStart, bgFirstImage);
    start = std::min(start, lastValidStart);
    int end = start + bgWindowSize - 1;

    // Print to console
    ui->Console->append(QString("<span style='color: cyan;'>Frame %1 → start: %2, end: %3</span>").arg(n).arg(start).arg(end));
    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
    ui->Console->append("<br>");

    // Generate parameter file
    //QString paramFilePath = QString("params_%1.txt").arg(n);
    QString outputDir = ui->BGpath->text();
    QDir dir(outputDir);
    // Create a "params" subdirectory
    if (!dir.exists("params")) {dir.mkdir("params");}
    // Use the "params" subdirectory for parameter files
    QDir paramDir(dir.filePath("params"));
    QString paramFilePath = paramDir.filePath(QString("bg_%1.param").arg(n, 4, 10, QChar('0')));

    bool useMedian = false;

    if (!BGCompute::generateParameterFile(paramFilePath,
                                          end, // numImages
                                          start, // starting image
                                          start, // bg start
                                          end,   // bg end
                                          inputPath,
                                          inputFileFormat,
                                          outputPath,
                                          useMedian))
    {
        ui->Console->append("<span style='color: red;'>Failed to create parameter file.</span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
        ui->Console->append("<br>");

        // Only schedule next frame if stop not requested
        if (!m_stopRequested) {
            processNextMovingBG(bgWindowSize, bgFirstImage, bgEndImage,
                                numImages, startImage,
                                inputPath, inputFileFormat, outputPath,
                                bgExePath, cudaPath, openCVPath);
        } else {
            ui->Console->append("<span style='color: orange;'><b>Moving background computation aborted.</b></span>");
            ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
            emit computeBGFinished(false); // emit a signal that the process has failed
        }

    }

    // Setup process
    QProcess* process = new QProcess(this);
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    #ifdef Q_OS_WIN
        if (!cudaPath.isEmpty())   env.insert("PATH", cudaPath + ";" + env.value("PATH"));
        if (!openCVPath.isEmpty()) env.insert("PATH", openCVPath + ";" + env.value("PATH"));
    #else
        if (!cudaPath.isEmpty())   env.insert("LD_LIBRARY_PATH", cudaPath + ":" + env.value("LD_LIBRARY_PATH"));
        if (!openCVPath.isEmpty()) env.insert("LD_LIBRARY_PATH", openCVPath + ":" + env.value("LD_LIBRARY_PATH"));
    #endif
    process->setProcessEnvironment(env);

    // Connect output/error
    connect(process, &QProcess::readyReadStandardOutput, this, [=]() {
        ui->Console->moveCursor(QTextCursor::End);
        ui->Console->insertPlainText(process->readAllStandardOutput());
        ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());
    });

    connect(process, &QProcess::readyReadStandardError, this, [=]() {
        ui->Console->moveCursor(QTextCursor::End);
        ui->Console->insertPlainText(process->readAllStandardError());
        ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());
    });

    // Connect finished signal
    connect(process, QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),
            this, [=](int exitCode, QProcess::ExitStatus /*status*/) {
                // Rename output background and determine success
                QString srcFile = outputPath + "background.png";
                QString dstFile = outputPath + QString("background_%1.png").arg(n, 4, 10, QChar('0'));

                if (QFile::exists(srcFile)) {
                    QFile::rename(srcFile, dstFile);
                    ui->Console->append("<span style='color: green;'>Done computing background for frame " + QString::number(n) + ".</span>");
                    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                    ui->Console->append("<br>");
                } else {
                    ui->Console->append("<span style='color: red;'>Background computation failed for frame " + QString::number(n) + ".</span>");
                    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                    ui->Console->append("<br>");
                    emit computeBGFinished(false); // emit a signal that the process has failed
                }
                process->deleteLater();

                // Start next frame
                // Start next frame if not stopped
                if (!m_stopRequested) {
                    processNextMovingBG(bgWindowSize, bgFirstImage, bgEndImage,
                                        numImages, startImage,
                                        inputPath, inputFileFormat, outputPath,
                                        bgExePath, cudaPath, openCVPath);
                } else {
                    ui->Console->append("<span style='color: orange;'><b>Moving background computation aborted.</b></span>");
                    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                    ui->Console->append("<br>");
                    emit computeBGFinished(false); // emit a signal that the process has failed
                }

            });

    // keep track of active processes, so that we can kill them using the stop button
    m_activeProcesses.append(process);

    // Remove from list when done
    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this, process](int, QProcess::ExitStatus){
                m_activeProcesses.removeAll(process);
                process->deleteLater();
            });


    // Start next frame if not stopped
    process->start(bgExePath, QStringList() << "-F" << paramFilePath);
}

// ================================================================== Functions for the Console page ==================================================================//
// stop button function to kill processes
void RIHVR::stopAllProcesses()
{
    m_stopRequested = true;   // prevent next frames from starting

    m_stopBatch = true; // stop the batch processing

    for (QProcess* proc : m_activeProcesses) {
        if (proc && proc->state() != QProcess::NotRunning) {
            proc->kill();
            proc->waitForFinished(3000);
        }
    }
    m_activeProcesses.clear();

    ui->Console->append("<span style='color: orange;'><b>All active processes terminated.</b></span>");
    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
    ui->Console->append("<br>");
}

// clear button function to clear console
void RIHVR::clearConsole()
{
    ui->Console->clear();
}

// save log file (console) function
void RIHVR::saveLogConsole()
{
    // Get the console text
    QString consoleText = ui->Console->toPlainText();

    // Get input image path from GUI
    QString inputPath = ui->ImgPath->text();
    if (!inputPath.endsWith('/')) inputPath.replace('\\','/') += '/';

    // Construct log file name with timestamp
    QString timeStamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    QString logFilePath = inputPath + "RIHVR_Log_" + timeStamp + ".log";

    // Save to file
    QFile file(logFilePath);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        out << consoleText;
        file.close();

        // Notify user in console
        ui->Console->append("<span style='color: green;'><b>Log saved to:</b> " + logFilePath + "</span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
        ui->Console->append("<br>");
    } else {
        ui->Console->append("<span style='color: red;'><b>Failed to save log file.</b></span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
        ui->Console->append("<br>");
        }
}

// ================================================================= Functions for the Hologram reconstruction page ==================================================================//
void RIHVR::computeHolographicReconstruction()
{
    // reset the stop button flag
    m_stopRequested = false; // reset at start

    // Switch to console tab
    ui->tabWidget->setCurrentIndex(4);

    ui->Console->append("<span style='color: cyan;'><b> Starting holographic reconstruction. </b/</span>");// Reset color/format to default for future messages
    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
    ui->Console->append("<br>");

    // Choose where to save the param file (temporary for now)

    // Gather GUI inputs
    int numImages      = 1;
    QString inputPath  = ui->HoloImgPath->text();
    QString inputFormat= ui->HoloImgFileFormat->text();
    QString outputPath = ui->OutputDir->text();
    double startZ      = ui->z0->text().toDouble();;
    double stepZ       = ui->dz->text().toDouble();
    int numPlanes      = ui->Nz->text().toInt();
    double wavelength  = ui->lambda->text().toDouble();
    double resolution  = ui->dx->text().toDouble();

    bool useROICenter  = ui->CenterRadio->isChecked();
    int roiX           = ui->x0->text().toInt();
    int roiY           = ui->y0->text().toInt();
    int roiSize        = ui->SquareSize->text().toInt();

    bool useROIRect    = ui->RectRadio->isChecked();
    int roiRectX       = ui->x0Rect->text().toInt();
    int roiRectY       = ui->y0Rect->text().toInt();
    int roiRectW       = ui->Lx->text().toInt();
    int roiRectH       = ui->Ly->text().toInt();

    int zeroPadding    = ui->ZeroPad->text().toInt();
    int invIter        = ui->nFISTA->text().toInt();
    double regTau      = ui->Sparsity->text().toDouble();
    double regTV       = ui->TV->text().toDouble();
    int tvIter         = ui->nTV->text().toInt();
    bool useEnhancement = !ui->ReconNoBG->isChecked();
    bool extractCentroids = ui->ExtractCentroids->isChecked();

    bool outputPlanes  = ui->PhaseEachPlaneCheck->isChecked();
    bool outputEachStep= ui->SaveFISTACheck->isChecked();

    // segmenation options
    int segMinIntensity = ui->MinVoxelIntensity->text().toInt();
    int segMinVoxelSz   = ui->MinVoxelSz->text().toInt();
    int segCloseSz      = ui->SegCloseSz->text().toInt();

    // check background type
    bool useNoBG     = ui->ReconNoBG->isChecked();
    bool useMedianBG = ui->ReconMedianBG->isChecked();
    bool useMovingBG = ui->ReconMovingBG->isChecked();
    bool useCCBG     = ui->ReconCorrelationBG->isChecked();

    QString bgPath   = ui->HoloBGpath->text();
    QString bgFile   = ui->BGName->text();
    QString bgFilename;


    if (!bgPath.isEmpty() && !bgFile.isEmpty()) {
        QDir dir(bgPath);
        bgFilename = dir.filePath(bgFile);  // automatically adds separator
    } else {
        bgFilename = "";  // leave empty if either part is missing
    }

    // === Check settings dialog ===
    if (!settingsdialog) {
        QMessageBox::warning(this, "Error", "Settings dialog not available.");
        return;
    }

    QString processExePath = settingsdialog->getProcessExePath();
    if (processExePath.isEmpty()) {
        QMessageBox::warning(this, "Error", "sparse-invere-recon.exe path not set. Please configure it in Settings.");
        emit computeHoloFinished(false); // emit a signal that the process has failed.
        return;
    }

    QString segmentationExePath = settingsdialog->getSegmentationExePath();
    if (processExePath.isEmpty()) {
        QMessageBox::warning(this, "Error", "segmentation.exe path not set. Please configure it in Settings.");
        emit computeHoloFinished(false); // emit a signal that the process has failed.
        return;
    }

    QString cudaPath   = settingsdialog->getCudaPath();
    QString openCVPath = settingsdialog->getOpenCVPath();

    // Generate parameter file
    QString outputDir = ui->OutputDir->text();
    QDir dir(outputDir);
    // ensures that the ouput directory exists
    if (!dir.exists()) {
        if (!dir.mkpath(".")) {  // "." means create the directory itself
            qWarning() << "Failed to create directory:" << outputPath;
        }
    }
    // Create a "params" subdirectory
    if (!dir.exists("params")) {dir.mkdir("params");}
    // Use the "params" subdirectory for parameter files
    QDir paramDir(dir.filePath("params"));

    int startImg = ui->HoloStartImage->text().toInt();
    int endImage   = ui->HoloEndImg->text().toInt();
    int increment  = ui->HololIncrement->text().toInt();

    // === Create frame queue ===
    m_HoloFrames.clear();
    for (int n = startImg; n <= endImage; n += increment)
        m_movingBGFrames.append(n);
     int medianWindow = 5; // it can't be changed in the computation, but we have it here to compute the correct number of image

    // Start processing first frame directly
    processNextHologram(startImg, endImage, increment,
                        startZ, stepZ, numPlanes,
                        wavelength, resolution,
                        inputPath, inputFormat, outputPath,
                        useROICenter, roiX, roiY, roiSize,
                        useROIRect, roiRectX, roiRectY, roiRectW, roiRectH,
                        segMinIntensity, segMinVoxelSz, segCloseSz,
                        zeroPadding, invIter, regTau, regTV, tvIter,
                        outputPlanes, outputEachStep,
                        useEnhancement, extractCentroids, useNoBG, useMedianBG, useMovingBG, useCCBG,
                        medianWindow,
                        bgPath, bgFile,
                        processExePath, segmentationExePath,
                        cudaPath, openCVPath);
}
//========= subroutine for hologram reconstruction function
void RIHVR::processNextHologram(int startImg, int endImg, int increment,
                                double startZ, double stepZ, int numPlanes,
                                double wavelength, double resolution,
                                const QString& inputPath, const QString& inputFileFormat,
                                const QString& outputPath,
                                bool useROICenter, int roiX, int roiY, int roiSize,
                                bool useROIRect, int roiRectX, int roiRectY, int roiRectW, int roiRectH,
                                int segMinIntensity, int segMinVoxelSz, int segCloseSz,
                                int zeroPadding, int invIter, double regTau, double regTV, int tvIter,
                                bool outputPlanes, bool outputEachStep,
                                bool useEnhancement, bool extractCentroids, bool useNoBG, bool useMedianBG, bool useMovingBG, bool useCCBG,
                                int medianWindow,
                                const QString& bgPath, const QString& bgPattern,
                                const QString& processExePath, const QString& segmentationExePath,
                                const QString& cudaPath, const QString& openCVPath)
{
    if (m_stopRequested || startImg > endImg) {
        ui->Console->append("<span style='color: green;'><b>Holographic reconstruction stopped or finished.</b></span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
        ui->Console->append("<br>");
        return;
    }

    int n = startImg;
    ui->Console->append("<span style='color: #9370DB;'>Processing image # " + QString::number(n) + "</span>");
    ui->Console->append("<span style='color: white;'></span>"); // reset color

    // Compute background filename
    QString bgFilename;

    // change the background name according to the image number (Moving)
    if (!useEnhancement || useNoBG) {
        // No enhancement or "No BG" selected
        bgFilename = "";
    } else {
        QString bgPath    = ui->HoloBGpath->text();
        QString bgPattern = ui->BGName->text();

        if (!bgPath.isEmpty() && !bgPattern.isEmpty()) {
            QDir dir(bgPath);

            if (useMovingBG || useCCBG) {
                // Moving BG and CCBG: expand current n with any %d/%0Nd pattern
                QRegularExpression re("%0?(\\d*)d");
                QRegularExpressionMatch match = re.match(bgPattern);

                QString patternForArg = bgPattern;
                int width = 0;

                if (match.hasMatch()) {
                    QString widthStr = match.captured(1);
                    if (!widthStr.isEmpty())
                        width = widthStr.toInt();
                    patternForArg.replace(re, "%1");
                }

                if (width > 0)
                    bgFilename = dir.filePath(patternForArg.arg(n, width, 10, QChar('0')));
                else
                    bgFilename = dir.filePath(patternForArg.arg(n));

            } else if (useMedianBG) {
                // Median BG: reuse last computed file if n is beyond available backgrounds
                int lastBGIndex = endImg - medianWindow + 1; // last available median BG
                int bgIndex = (n > lastBGIndex) ? lastBGIndex : n;

                QRegularExpression re("%0?(\\d*)d");
                QRegularExpressionMatch match = re.match(bgPattern);

                QString patternForArg = bgPattern;
                int width = 0;
                if (match.hasMatch()) {
                    QString widthStr = match.captured(1);
                    if (!widthStr.isEmpty())
                        width = widthStr.toInt();
                    patternForArg.replace(re, "%1");
                }

                if (width > 0)
                    bgFilename = dir.filePath(patternForArg.arg(bgIndex, width, 10, QChar('0')));
                else
                    bgFilename = dir.filePath(patternForArg.arg(bgIndex));

            } else {
                // Mean/static BG: just use the filename as-is
                bgFilename = dir.filePath(bgPattern);
            }
        } else {
            bgFilename = ""; // fallback
        }
    }

    // Generate parameter file in OutputDir/params/
    QDir outDir(outputPath);
    if (!outDir.exists("params")) outDir.mkdir("params");
    QDir paramDir(outDir.filePath("params"));
    QString paramFilePath = paramDir.filePath(QString("reconstruction_%1.param").arg(n, 4, 10, QChar('0')));

    bool ok = ReconstructionCompute::generateParameterFile(
        paramFilePath,
        1,            // numImages per param file
        n,            // startImage
        inputPath, inputFileFormat, outputPath,
        startZ, stepZ, numPlanes,
        wavelength, resolution,
        useROICenter, roiX, roiY, roiSize,
        useROIRect, roiRectX, roiRectY, roiRectW, roiRectH,
        segMinIntensity, segMinVoxelSz, segCloseSz,
        zeroPadding, invIter,
        regTau, regTV, tvIter,
        outputPlanes, outputEachStep,
        useEnhancement, extractCentroids, bgFilename
        );

    if (!ok) {
        ui->Console->append("<span style='color: red;'>Failed to create parameter file for frame " + QString::number(n) + "</span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
        ui->Console->append("<br>");
        emit computeHoloFinished(false); // emit a signal that the process has failed.
        return;
    }

    // Setup QProcess
    QProcess* process = new QProcess(this);
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    #ifdef Q_OS_WIN
        if (!cudaPath.isEmpty())   env.insert("PATH", cudaPath + ";" + env.value("PATH"));
        if (!openCVPath.isEmpty()) env.insert("PATH", openCVPath + ";" + env.value("PATH"));
    #else
        if (!cudaPath.isEmpty())   env.insert("LD_LIBRARY_PATH", cudaPath + ":" + env.value("LD_LIBRARY_PATH"));
        if (!openCVPath.isEmpty()) env.insert("LD_LIBRARY_PATH", openCVPath + ":" + env.value("LD_LIBRARY_PATH"));
    #endif
    process->setProcessEnvironment(env);

    // Connect output/error to console
    connect(process, &QProcess::readyReadStandardOutput, this, [=]() {
        ui->Console->moveCursor(QTextCursor::End);
        ui->Console->insertPlainText(process->readAllStandardOutput());
        ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());
    });
    connect(process, &QProcess::readyReadStandardError, this, [=]() {
        ui->Console->moveCursor(QTextCursor::End);
        ui->Console->insertPlainText(process->readAllStandardError());
        ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());
    });

    // Handle process finished
    connect(process, QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),
            this, [=](int /*exitCode*/, QProcess::ExitStatus /*status*/) {

        // Expected output file
        QString outFile = QDir(outputPath).filePath("estimate_final.tif");

        if (QFileInfo::exists(outFile)) {
            ui->Console->append("<span style='color: green;'>Finished reconstructing frame " + QString::number(n) + "</span>");
            ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages

            //===============Rename and move voxel data, projections, and enchanced images. if desired

            // move enchanced image and rename it (given that we want to save it)
            if (ui->EnhanceCheck->isChecked()) {
                QString EnhanceFile = outDir.filePath("enh.tif");
                if (QFileInfo::exists(EnhanceFile)) {
                    QDir EnhDir(outDir.filePath("Enhanced"));
                    if (!EnhDir.exists()) EnhDir.mkpath(".");

                    QString newEnhName = EnhDir.filePath(QString("enh_%1.tif").arg(n, 4, 10, QChar('0')));
                    // If destination already exists, remove it first
                    if (QFileInfo::exists(newEnhName)) {
                        QFile::remove(newEnhName);
                    }
                    if (!QFile::rename(EnhanceFile, newEnhName)) {
                        ui->Console->append("<span style='color: red;'>Failed to move enhanced image: " + EnhanceFile + "</span>");
                        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                    }
                } else {
                    ui->Console->append("<span style='color: orange;'>Enhanced image does not exist for frame " + QString::number(n) + "</span>");
                    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                }
            }


            // move XY projection file and rename it (given that we want to save it)
            if (ui->PhaseProjCheck->isChecked()) {
                QString XYFile = outDir.filePath("cmb_xy.tif");
                if (QFileInfo::exists(XYFile)) {
                    QDir ProjDir(outDir.filePath("Projections"));
                    if (!ProjDir.exists()) ProjDir.mkpath(".");

                    QString newXYName = ProjDir.filePath(QString("cmb_xy_%1.tif").arg(n, 4, 10, QChar('0')));
                    // If destination already exists, remove it first
                    if (QFileInfo::exists(newXYName)) {
                        QFile::remove(newXYName);
                    }
                    if (!QFile::rename(XYFile, newXYName)) {
                        ui->Console->append("<span style='color: red;'>Failed to move XY projection data: " + XYFile + "</span>");
                        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                    }
                } else {
                    ui->Console->append("<span style='color: orange;'>XY projection file does not exist for frame " + QString::number(n) + "</span>");
                    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                }
            }

            // move XZ projection file and rename it (given that we want to save it)
            if (ui->PhaseProjCheck->isChecked()) {
                QString XZFile = outDir.filePath("cmb_xz.tif");
                if (QFileInfo::exists(XZFile)) {
                    QDir ProjDir(outDir.filePath("Projections"));
                    if (!ProjDir.exists()) ProjDir.mkpath(".");

                    QString newXZName = ProjDir.filePath(QString("cmb_xz_%1.tif").arg(n, 4, 10, QChar('0')));
                    // If destination already exists, remove it first
                    if (QFileInfo::exists(newXZName)) {
                        QFile::remove(newXZName);
                    }
                    if (!QFile::rename(XZFile, newXZName)) {
                        ui->Console->append("<span style='color: red;'>Failed to move XZ projection data: " + XZFile + "</span>");
                        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                    }
                } else {
                    ui->Console->append("<span style='color: orange;'>XZ projection file does not exist for frame " + QString::number(n) + "</span>");
                    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                }
            }

            // move YZ projection file and rename it (given that we want to save it)
            if (ui->PhaseProjCheck->isChecked()) {
                QString YZFile = outDir.filePath("cmb_yz.tif");
                if (QFileInfo::exists(YZFile)) {
                    QDir ProjDir(outDir.filePath("Projections"));
                    if (!ProjDir.exists()) ProjDir.mkpath(".");

                    QString newYZName = ProjDir.filePath(QString("cmb_yz_%1.tif").arg(n, 4, 10, QChar('0')));
                    // If destination already exists, remove it first
                    if (QFileInfo::exists(newYZName)) {
                        QFile::remove(newYZName);
                    }
                    if (!QFile::rename(YZFile, newYZName)) {
                        ui->Console->append("<span style='color: red;'>Failed to move YZ projection data: " + YZFile + "</span>");
                        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                        }
                } else {
                        ui->Console->append("<span style='color: orange;'>YZ projection file does not exist for frame " + QString::number(n) + "</span>");
                        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                }
            }

            // rename the voxel data file
            QString voxelFile = outDir.filePath("inverse_voxels.csv");
            QString newVoxelName = outDir.filePath(QString("inverse_voxels_%1.csv").arg(n, 4, 10, QChar('0')));
            QFile::remove(newVoxelName); // remove the file, if the same file name already exists
            if (!QFile::rename(voxelFile, newVoxelName)) { // Rename in-place
                ui->Console->append("<span style='color: red;'>Failed to rename voxel data: " + voxelFile + "</span>");
                ui->Console->append("<span style='color: white;'></span>");
            }

            // run segmentation to get centroids if the extract centroid flag is on
            if (ui->ExtractCentroids->isChecked()){
                ui->Console->append("<span style='color: cyan;'>Extracting centroids from: " + voxelFile + "</span>");
                ui->Console->append("<span style='color: white;'></span>");
            }

            // run segmentation to get centroids if the ExtractCentroids flag is on
            if (ui->ExtractCentroids->isChecked()) {
                QProcess* segProcess = new QProcess(this);
                QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
                #ifdef Q_OS_WIN
                                if (!cudaPath.isEmpty())   env.insert("PATH", cudaPath + ";" + env.value("PATH"));
                                if (!openCVPath.isEmpty()) env.insert("PATH", openCVPath + ";" + env.value("PATH"));
                #else
                                if (!cudaPath.isEmpty())   env.insert("LD_LIBRARY_PATH", cudaPath + ":" + env.value("LD_LIBRARY_PATH"));
                                if (!openCVPath.isEmpty()) env.insert("LD_LIBRARY_PATH", openCVPath + ":" + env.value("LD_LIBRARY_PATH"));
                #endif
                // set environment if needed
                segProcess->setProcessEnvironment(env);

                // Connect console output
                connect(segProcess, &QProcess::readyReadStandardOutput, this, [=]() {
                    ui->Console->moveCursor(QTextCursor::End);
                    ui->Console->insertPlainText(segProcess->readAllStandardOutput());
                    ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());
                });
                connect(segProcess, &QProcess::readyReadStandardError, this, [=]() {
                    ui->Console->moveCursor(QTextCursor::End);
                    ui->Console->insertPlainText(segProcess->readAllStandardError());
                    ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());
                });

                // Handle finished
                connect(segProcess, QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),
                        this, [=](int exitCode, QProcess::ExitStatus) {
                            if (exitCode == 0)
                                {ui->Console->append("<span style='color: green;'>Centroid extraction finished successfully.</span>");
                                ui->Console->append("<span style='color: white;'></span>");

                                // move the weighted centroids to the centroids folder
                                QString centroidwFile = outDir.filePath(QString("centroids_weighted_%1.csv").arg(n, 4, 10, QChar('0')));
                                if (QFileInfo::exists(centroidwFile)) {
                                    QDir centroidwDir(outDir.filePath("Centroids"));
                                    if (!centroidwDir.exists()) centroidwDir.mkpath(".");

                                    QString newCentroidwName = centroidwDir.filePath(QString("centroids_weighted_%1.csv").arg(n, 4, 10, QChar('0')));
                                    // If destination already exists, remove it first
                                    if (QFileInfo::exists(newCentroidwName)) {
                                        QFile::remove(newCentroidwName);
                                    }

                                    if (!QFile::rename(centroidwFile, newCentroidwName)) {
                                        ui->Console->append("<span style='color: red;'>Failed to move weighted centroid file: " + centroidwFile + "</span>");
                                        ui->Console->append("<span style='color: white;'></span>");
                                    }
                                } else {
                                    ui->Console->append("<span style='color: orange;'>Weighted centroid data file does not exist for frame " + QString::number(n) + "</span>");
                                    ui->Console->append("<span style='color: white;'></span>");
                                }

                                // move the verbose centroids to the centroids folder
                                QString centroidvFile = outDir.filePath(QString("centroids_verbose_%1.csv").arg(n, 4, 10, QChar('0')));
                                if (QFileInfo::exists(centroidvFile)) {
                                    QDir centroidvDir(outDir.filePath("Centroids"));
                                    if (!centroidvDir.exists()) centroidvDir.mkpath(".");

                                    QString newCentroidvName = centroidvDir.filePath(QString("centroids_verbose_%1.csv").arg(n, 4, 10, QChar('0')));
                                    // If destination already exists, remove it first
                                    if (QFileInfo::exists(newCentroidvName)) {
                                        QFile::remove(newCentroidvName);
                                    }

                                    if (!QFile::rename(centroidvFile, newCentroidvName)) {
                                        ui->Console->append("<span style='color: red;'>Failed to move verbose centroid file: " + centroidvFile + "</span>");
                                        ui->Console->append("<span style='color: white;'></span>");
                                    }
                                } else {
                                    ui->Console->append("<span style='color: orange;'>Verbose centroid data file does not exist for frame " + QString::number(n) + "</span>");
                                    ui->Console->append("<span style='color: white;'></span>");
                                }

                                // move the centroids to the centroids folder
                                QString centroidFile = outDir.filePath(QString("centroids_%1.csv").arg(n, 4, 10, QChar('0')));
                                    if (QFileInfo::exists(centroidFile)) {
                                        QDir centroidDir(outDir.filePath("Centroids"));
                                        if (!centroidDir.exists()) centroidDir.mkpath(".");

                                        QString newCentroidName = centroidDir.filePath(QString("centroids_%1.csv").arg(n, 4, 10, QChar('0')));
                                        // If destination already exists, remove it first
                                        if (QFileInfo::exists(newCentroidName)) {
                                            QFile::remove(newCentroidName);
                                        }

                                        if (!QFile::rename(centroidFile, newCentroidName)) {
                                            ui->Console->append("<span style='color: red;'>Failed to move centroid file: " + centroidFile + "</span>");
                                            ui->Console->append("<span style='color: white;'></span>");
                                        }
                                } else {
                                    ui->Console->append("<span style='color: orange;'>Centroid data file does not exist for frame " + QString::number(n) + "</span>");
                                    ui->Console->append("<span style='color: white;'></span>");
                                }

                                // move voxel file and rename it (given that we want to save it)
                                QString voxelFile = outDir.filePath(QString("inverse_voxels_%1.csv").arg(n, 4, 10, QChar('0')));
                                if (ui->VoxelDataCheck->isChecked()) {
                                    if (QFileInfo::exists(voxelFile)) {
                                        QDir voxelDir(outDir.filePath("VoxelData"));
                                        if (!voxelDir.exists()) voxelDir.mkpath(".");

                                        QString newVoxelName = voxelDir.filePath(QString("inverse_voxels_%1.csv").arg(n, 4, 10, QChar('0')));
                                        // If destination already exists, remove it first
                                        if (QFileInfo::exists(newVoxelName)) {
                                            QFile::remove(newVoxelName);
                                        }

                                        if (!QFile::rename(voxelFile, newVoxelName)) {
                                            ui->Console->append("<span style='color: red;'>Failed to move voxel data: " + voxelFile + "</span>");
                                            ui->Console->append("<span style='color: white;'></span>");
                                        }
                                    } else {
                                        ui->Console->append("<span style='color: orange;'>Voxel data file does not exist for frame " + QString::number(n) + "</span>");
                                        ui->Console->append("<span style='color: white;'></span>");
                                    }
                                } else {
                                    // otherwise, delete the file if it exists
                                    if (QFileInfo::exists(voxelFile)) {
                                        if (!QFile::remove(voxelFile)) {
                                            ui->Console->append("<span style='color: red;'>Failed to delete voxel data: " + voxelFile + "</span>");
                                        } else {
                                            ui->Console->append("<span style='color: gray;'>Deleted voxel data: " + voxelFile + "</span>");
                                        }
                                        ui->Console->append("<span style='color: white;'></span>");
                                    }
                                }

                                if (n == endImg) {
                                    ui->Console->append("<span style='color: green;'><b>All frames processed successfully.</b></span>");
                                    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                                    ui->Console->append("<br>");
                                    emit computeHoloFinished(true); // emit a signal that the process has finised successfully.
                                }
                            }
                            else

                                {ui->Console->append("<span style='color: red;'>Segmentation failed with code " + QString::number(exitCode) + "</span>");
                                ui->Console->append("<span style='color: white;'></span>");
                                emit computeHoloFinished(false); // emit a signal that the process has failed.
                                }
                            });
                    // Start the segmentation process
                    segProcess->start(segmentationExePath, QStringList()<< "-F" << paramFilePath);
            }else{
                // move voxel file and rename it (given that we want to save it)
                QString voxelFile = outDir.filePath(QString("inverse_voxels_%1.csv").arg(n, 4, 10, QChar('0')));
                if (ui->VoxelDataCheck->isChecked()) {
                    if (QFileInfo::exists(voxelFile)) {
                        QDir voxelDir(outDir.filePath("VoxelData"));
                        if (!voxelDir.exists()) voxelDir.mkpath(".");

                        QString newVoxelName = voxelDir.filePath(QString("inverse_voxels_%1.csv").arg(n, 4, 10, QChar('0')));
                        // If destination already exists, remove it first
                        if (QFileInfo::exists(newVoxelName)) {
                            QFile::remove(newVoxelName);
                        }

                        if (!QFile::rename(voxelFile, newVoxelName)) {
                            ui->Console->append("<span style='color: red;'>Failed to move voxel data: " + voxelFile + "</span>");
                            ui->Console->append("<span style='color: white;'></span>");
                        }
                    } else {
                        ui->Console->append("<span style='color: orange;'>Voxel data file does not exist for frame " + QString::number(n) + "</span>");
                        ui->Console->append("<span style='color: white;'></span>");
                    }
                } else {
                    // otherwise, delete the file if it exists
                    if (QFileInfo::exists(voxelFile)) {
                        if (!QFile::remove(voxelFile)) {
                            ui->Console->append("<span style='color: red;'>Failed to delete voxel data: " + voxelFile + "</span>");
                        } else {
                            ui->Console->append("<span style='color: gray;'>Deleted voxel data: " + voxelFile + "</span>");
                        }
                        ui->Console->append("<span style='color: white;'></span>");
                    }
                }
            }

            if (n == endImg && !(ui->ExtractCentroids->isChecked())) {
                ui->Console->append("<span style='color: green;'><b>All frames processed successfully.</b></span>");
                ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                ui->Console->append("<br>");

                emit computeHoloFinished(true); // emit a signal that the process has finised successfully.
            }
            //ui->Console->append("<br>");
        }

        else {
                ui->Console->append("<span style='color: red;'>Frame " + QString::number(n) + " failed. Couldn't find estimate_final.tif. </span>");
                ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
                ui->Console->append("<br>");
                }

        m_activeProcesses.removeAll(process);
        process->deleteLater();

        // Queue next frame
        int nextN = n + increment;
        if (!m_stopRequested && nextN <= endImg) {
            processNextHologram(nextN, endImg, increment,startZ, stepZ, numPlanes,wavelength, resolution,
                                        inputPath, inputFileFormat, outputPath,useROICenter, roiX, roiY, roiSize,
                                        useROIRect, roiRectX, roiRectY, roiRectW, roiRectH, segMinIntensity, segMinVoxelSz, segCloseSz,
                                        zeroPadding, invIter, regTau, regTV, tvIter,outputPlanes, outputEachStep,
                                        useEnhancement, extractCentroids, useNoBG, useMedianBG, useMovingBG,useCCBG, medianWindow,
                                        bgPath, bgPattern,processExePath,segmentationExePath,cudaPath, openCVPath);
        }
        else if (m_stopRequested) {
            ui->Console->append("<span style='color: orange;'><b>Reconstruction aborted by user.</b></span>");
            ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
            ui->Console->append("<br>");
            emit computeHoloFinished(false); // emit a signal that the process has failed.
        }
        else {
            // Set flag instead of printing
            AllDoneFlag = true;
            //ui->Console->append("<span style='color: green;'><b>All frames processed successfully.</b></span>");
            //ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
            //ui->Console->append("<br>");

        }
    });

    int SubFolderFlag = ui->SubfoldersCheck->isChecked() ? 1 : 0; // check is saving to a subfolder or not

    m_activeProcesses.append(process);
    process->start(processExePath, QStringList()<< "-F" << paramFilePath << "-S" << QString::number(SubFolderFlag));
}

// ================================================================== Tracking page =======================================================================//

// this function is for the correlation-based background computation
void RIHVR::computeTracks()
{
    // reset the stop button flag
    m_stopRequested = false; // reset at start

    // switch to console
    ui->tabWidget->setCurrentIndex(4);

    // Clear the console at the start
    //ui->Console->clear();
    ui->Console->append("<span style='color: cyan;'><b>Computing particle tracks...</b></span>");
    ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages

    // Collect GUI values
    QString inputPath       = ui->CentroidDataPath->text();
    QString inputFileFormat = ui->CentroidDataFileFormat->text();
    QString outputPath = ui->TrackDirPath->text();
    int trackStartIdX = ui->TrackStartIdX->text().toInt();
    int trackEndIdX = ui->TrackEndIdX->text().toInt();
    int maxDisp = ui->MaxDisplacement->text().toInt();
    int memory = ui->ParticleMemory->text().toInt();
    int minTrajLength = ui->MinTrajLength->text().toInt();
    int minParticleSize = ui->MinParticleSize->text().toInt();
    int maxParticleSize = ui->MaxParticleSize->text().toInt();
    int dimensions = ui->Dimensions->text().toInt();
    bool quietFlag = !(ui->Print2Console->isChecked());

    // Normalize paths
    if (!inputPath.endsWith('/'))  inputPath.replace('\\','/') += '/';
    if (!outputPath.endsWith('/')) outputPath.replace('\\','/') += '/';

    //QString paramFilePath = "params.txt";
    QString outputDir = ui->TrackDirPath->text();
    QDir dir(outputDir);
    // ensures that the ouput directory exists
    if (!dir.exists()) {
        if (!dir.mkpath(".")) {  // "." means create the directory itself
            qWarning() << "Failed to create directory:" << outputPath;
        }
    }

    QDir paramDir(dir.filePath(""));
    QString paramFilePath = paramDir.filePath(QString("tracks.params"));

    // Generate parameter file
    if (!TrackingCompute::generateParameterFile(paramFilePath,
                                                trackStartIdX, trackEndIdX,
                                                maxParticleSize, minParticleSize,
                                                maxDisp, minTrajLength,
                                                memory, dimensions,
                                                inputPath, inputFileFormat,
                                                outputPath, quietFlag))
    {
        QMessageBox::critical(this, "Error", "Failed to create parameter file.");
        return;
    }

    // Check settings dialog
    if (!settingsdialog) {
        QMessageBox::warning(this, "Error", "Settings dialog not available.");
        return;
    }

    QString trackingExePath = settingsdialog->getTrackingExePath();
    if (trackingExePath.isEmpty()) {
        QMessageBox::warning(this, "Error", "particletracking.exe path is not set. Please configure it in Settings.");
        return;
    }

    // Get CUDA/OpenCV paths
    QString cudaPath   = settingsdialog->getCudaPath();
    QString openCVPath = settingsdialog->getOpenCVPath();

    // Create process
    QProcess* process = new QProcess(this);

    // Set environment
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    #ifdef Q_OS_WIN
        if (!cudaPath.isEmpty())   env.insert("PATH", cudaPath + ";" + env.value("PATH"));
        if (!openCVPath.isEmpty()) env.insert("PATH", openCVPath + ";" + env.value("PATH"));
    #else
        if (!cudaPath.isEmpty())   env.insert("LD_LIBRARY_PATH", cudaPath + ":" + env.value("LD_LIBRARY_PATH"));
        if (!openCVPath.isEmpty()) env.insert("LD_LIBRARY_PATH", openCVPath + ":" + env.value("LD_LIBRARY_PATH"));
    #endif
    process->setProcessEnvironment(env);

    // Connect signals for real-time output
    connect(process, &QProcess::readyReadStandardOutput, this, [this, process]() {
        QString output = process->readAllStandardOutput();
        ui->Console->moveCursor(QTextCursor::End);
        ui->Console->insertPlainText(output);
        ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());

    });

    connect(process, &QProcess::readyReadStandardError, this, [this, process]() {
        QString error = process->readAllStandardError();
        ui->Console->moveCursor(QTextCursor::End);
        ui->Console->insertPlainText(error);
        ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());
    });

    // Handle process finished
    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this, outputPath](int exitCode, QProcess::ExitStatus exitStatus) {
                bool success = false;

                if (exitStatus == QProcess::NormalExit && exitCode == 0)
                    success = true;

                if (success) {
                    ui->Console->append("<span style='color: green;'><b>Particle tracking computation completed successfully.</b></span>");
                    emit computeTracksFinished(true); // emit a signal that the process has finised successfully.

                    //QMessageBox::information(this, "Particle Tracking", "Completed successfully.");
                } else {
                    QString errMsg = QString("Particle tracking computation failed (exit code %1).").arg(exitCode);
                    ui->Console->append("<span style='color: red;'><b>" + errMsg + "</b></span>");
                    QMessageBox::critical(this, "Particle Tracking",
                                          "Failed to finish particle tracking computation.\nCheck output folder or console output.");
                     emit computeTracksFinished(false); // emit a signal that the process has stopped.
                }

                ui->Console->append("<span style='color: white;'></span>");
            });



    // keep track of active processes, so that we can kill them using the stop button
    m_activeProcesses.append(process);

    // Remove from list when done
    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this, [this, process](int, QProcess::ExitStatus){
                m_activeProcesses.removeAll(process);
                process->deleteLater();
            });

    // Start the process
    //qDebug() << "Launching:" << trackingExePath << paramFilePath;
    process->start(trackingExePath, QStringList() << paramFilePath);
    if (!process->waitForStarted()) {
        ui->Console->append("<span style='color: red;'><b>Failed to start particletracking.exe.</b></span>");
        ui->Console->append("<span style='color: white;'></span>");// Reset color/format to default for future messages
        QMessageBox::critical(this, "Error", "Failed to start particletracking.exe");
        return;
    }
}



// ================================================================== Visualization page =======================================================================//
// function to update the hologram visualiation path
void RIHVR::updateHoloVizPath()
{
    static bool localSync = false;
    if (localSync) return;
    localSync = true;

    QString folder = ui->HoloImgPath->text().trimmed();
    QString format = ui->HoloImgFileFormat->text().trimmed();

    bool ok1, ok2, ok3;
    int startNum = ui->HoloStartImage->text().toInt(&ok1);
    int endNum   = ui->HoloEndImg->text().toInt(&ok2);
    int inc      = ui->HololIncrement->text().toInt(&ok3);
    if (!ok1 || !ok2 || !ok3 || inc <= 0) inc = 1;  // default if invalid

    // Compute number of steps
    int totalSteps = ((endNum - startNum) / inc);
    if (totalSteps < 0) totalSteps = 0;

    // Update slider range dynamically
    ui->HoloSlider->setMinimum(0);
    ui->HoloSlider->setMaximum(totalSteps);

    // Current slider position → compute actual image number
    int sliderPos = ui->HoloSlider->value();
    int currentNum = startNum + sliderPos * inc;

    if (folder.isEmpty() || format.isEmpty()) {
        ui->HoloVizImgPath->setText("");
        localSync = false;
        return;
    }

    // Ensure folder ends with slash
    if (!folder.endsWith("/") && !folder.endsWith("\\"))
        folder += "/";

    // Format the filename using sprintf-style format
    QString filename = QString::asprintf(format.toStdString().c_str(), currentNum);

    ui->HoloVizImgPath->setText(folder + filename);

    localSync = false;
}

// function to update the max phase projection visualiation path
void RIHVR::updatePhaseVizPath()
{
    static bool localSync = false;
    if (localSync) return;
    localSync = true;

    // Base folder
    QString folder = ui->OutputDir->text().trimmed();

    // Append 'Projections' subfolder
    if (!folder.endsWith("/") && !folder.endsWith("\\"))
        folder += "/";
    folder += "Projections/";

    // Determine projection type
    QString format;
    if (ui->XYProj->isChecked())
        format = "cmb_xy_%04d.tif";
    else if (ui->YZProj->isChecked())
        format = "cmb_yz_%04d.tif";
    else if (ui->XZProj->isChecked())
        format = "cmb_xz_%04d.tif";
    else {
        ui->PhaseVizImgPath->setText("");
        localSync = false;
        return;
    }

    // Get range and increment
    bool ok1, ok2, ok3;
    int startNum = ui->HoloStartImage->text().toInt(&ok1);
    int endNum   = ui->HoloEndImg->text().toInt(&ok2);
    int inc      = ui->HololIncrement->text().toInt(&ok3);
    if (!ok1 || !ok2 || !ok3 || inc <= 0) inc = 1;

    // Compute total steps
    int totalSteps = ((endNum - startNum) / inc);
    if (totalSteps < 0) totalSteps = 0;

    // Update slider range
    ui->PhaseSlider->setMinimum(0);
    ui->PhaseSlider->setMaximum(totalSteps);

    // Compute current number
    int sliderPos = ui->PhaseSlider->value();
    int currentNum = startNum + sliderPos * inc;

    // Format filename and update
    QString filename = QString::asprintf(format.toStdString().c_str(), currentNum);
    ui->PhaseVizImgPath->setText(folder + filename);

    localSync = false;
}

// to sync holo and phase sliders
void RIHVR::setupSliderSync()
{
    // When HoloSlider changes
    connect(ui->HoloSlider, &QSlider::valueChanged, this, [this](int value) {
        bool sync = ui->SyncWithHolo->isChecked();

        if (sync) {
            QSignalBlocker b(ui->PhaseSlider);
            ui->PhaseSlider->setValue(value);
        }

        updateHoloVizPath();
        if (sync) updatePhaseVizPath();  // also update phase view
    });

    // When PhaseSlider changes
    connect(ui->PhaseSlider, &QSlider::valueChanged, this, [this](int value) {
        bool sync = ui->SyncWithHolo->isChecked();

        if (sync) {
            QSignalBlocker b(ui->HoloSlider);
            ui->HoloSlider->setValue(value);
        }

        updatePhaseVizPath();
        if (sync) updateHoloVizPath();  // also update holo view
    });
}

// browse button for holovizimage
void RIHVR::on_HoloVizImgBrowseButton_clicked()
{
    // Ask user to select a sample hologram image
    QString sampleFile = QFileDialog::getOpenFileName(
        this,
        tr("Select Sample Hologram Image"),
        ui->HoloImgPath->text(),
        tr("Images (*.tif *.tiff *.png *.jpg *.bmp)")
        );

    if (sampleFile.isEmpty())
        return;

    QFileInfo fileInfo(sampleFile);
    QString folderPath = fileInfo.absolutePath();
    QString fileName = fileInfo.fileName(); // e.g. "Holo_00001234.tif"
    QString suffix = fileInfo.suffix();

    // Always use forward slashes and ensure trailing slash
    folderPath.replace("\\", "/");
    if (!folderPath.endsWith('/'))
        folderPath += '/';

    // Extract numeric sequence info from filename
    QRegularExpression lastNumRx("(\\d+)(?!.*\\d)");
    QRegularExpressionMatch m = lastNumRx.match(fileName);
    if (!m.hasMatch()) {
        QMessageBox::warning(this, tr("Error"), tr("No numeric part found in filename."));
        return;
    }

    QString numberStr = m.captured(1);       // e.g. "00001234"
    int numDigits = numberStr.length();
    int currentIndex = numberStr.toInt();
    int startIndex = m.capturedStart(1);
    QString prefix = fileName.left(startIndex);
    QString extension = suffix;

    // Build a printf-style format string safely (e.g. "Holo_%08d.tif")
    QString filePattern = prefix + "%" + QString("0%1d").arg(numDigits) + "." + extension;

    // Scan the directory for min/max indices with same prefix+extension
    QDir dir(folderPath);
    QString nameFilter = prefix + "*" + "." + extension;
    dir.setNameFilters({ nameFilter });

    QStringList fileList = dir.entryList(QDir::Files, QDir::Name);
    int minNum = currentIndex;
    int maxNum = currentIndex;

    QRegularExpression numRx("(\\d+)(?!.*\\d)");
    for (const QString &fn : fileList) {
        QRegularExpressionMatch mm = numRx.match(fn);
        if (mm.hasMatch()) {
            int n = mm.captured(1).toInt();
            if (n < minNum) minNum = n;
            if (n > maxNum) maxNum = n;
        }
    }

    // Update the processing tab fields
    ui->HoloImgPath->setText(folderPath);
    ui->HoloImgFileFormat->setText(filePattern);
    ui->HoloStartImage->setText(QString::number(minNum));
    ui->HoloEndImg->setText(QString::number(maxNum));

    //Update visualization path too (same as Holo path initially)
    ui->HoloVizImgPath->setText(sampleFile);

    // reset the Holo slider to the current frame
    ui->HoloSlider->setValue(currentIndex);

}

// helper function to update both images
void RIHVR::updateHoloImageDisplay()
{
    QString imgPath = ui->HoloVizImgPath->text().trimmed();
    if (imgPath.isEmpty())
        return;

    // Get user-defined contrast range (0–1)
    double lo = ui->ContrastLO->value();
    double hi = ui->ContrastHI->value();

    // Load main holo image
    cv::Mat holoMat = VizUtils::loadImageAsMat(imgPath);
    if (holoMat.empty()) {
        ui->HoloImgViz->setText("Failed to load image");
        ui->HoloImgViz->setAlignment(Qt::AlignCenter);
        return;
    }

    // Handle background subtraction if requested
    if (ui->HoloVizSubtractBG->isChecked()) {

        cv::Mat bgMat;
        QString bgFolder = ui->HoloBGpath->text().trimmed();
        bool bgFound = true;
        QString bgPattern = ui->BGName->text().trimmed();

        if (ui->ReconNoBG->isChecked()) {
            bgFound = false;
            ui->Console->append("No background selected for subtraction.");
        }
        else if (ui->ReconMeanBG->isChecked()) {
            // mean background: just use the filename literally
            QString bgFile = QDir(bgFolder).filePath(bgPattern);
            if (!QFileInfo::exists(bgFile)) {
                bgFound = false;
                ui->Console->append(QString("Mean BG selected but file not found: %1").arg(bgFile));
            } else {
                bgMat = VizUtils::loadImageAsMat(bgFile);
            }
        }
        else if (ui->ReconMovingBG->isChecked() || ui->ReconCorrelationBG->isChecked()) {
            int idx = ui->HoloSlider->value();
            bgMat = VizUtils::loadBackgroundImage(bgFolder, bgPattern, idx);
            if (bgMat.empty()) {
                bgFound = false;
                ui->Console->append(QString("BG file not found for index %1 using pattern %2 in folder %3")
                                        .arg(idx).arg(bgPattern).arg(bgFolder));
            }
        }
        else if (ui->ReconMedianBG->isChecked()) {
            int idx = ui->HoloSlider->value();
            bgMat = VizUtils::loadBackgroundImage(bgFolder, bgPattern, idx, true); // pickLargestIfMissing = true
            if (bgMat.empty()) {
                bgFound = false;
                ui->Console->append(QString("Median BG file not found in folder %1 with pattern %2")
                                        .arg(bgFolder).arg(bgPattern));
            }
        }

        // Subtract background if found
        if (bgFound && !bgMat.empty()) {
            // Ensure size/channels match
            if (bgMat.size() != holoMat.size() || bgMat.type() != holoMat.type()) {
                cv::resize(bgMat, bgMat, holoMat.size());
                if (bgMat.channels() != holoMat.channels()) {
                    if (holoMat.channels() == 1)
                        cv::cvtColor(bgMat, bgMat, cv::COLOR_BGR2GRAY);
                    else
                        cv::cvtColor(bgMat, bgMat, cv::COLOR_GRAY2BGR);
                }
            }

            cv::Mat diff;
            // subtract and convert to float to allow negatives
            holoMat.convertTo(diff, CV_32F);
            cv::Mat bgFloat;
            bgMat.convertTo(bgFloat, CV_32F);
            diff -= bgFloat;

            // normalize to 0-255
            cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX);

            // back to 8-bit for display
            diff.convertTo(holoMat, CV_8U);
        }
    }

    // contrast scaling
    // Convert holoMat to float for display scaling
    cv::Mat displayMat;
    holoMat.convertTo(displayMat, CV_32F);

    // If LO==0 and HI==1, the range is full — default
    double minVal, maxVal;
    cv::minMaxLoc(displayMat, &minVal, &maxVal);

    // Scale: lo -> 0, hi -> 255
    displayMat = (displayMat - (minVal + lo * (maxVal - minVal))) * (255.0 / ((hi - lo) * (maxVal - minVal)));

    // Clip values to 0–255
    cv::threshold(displayMat, displayMat, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(displayMat, displayMat, 0, 0, cv::THRESH_TOZERO);

    // Convert to 8-bit for QImage
    cv::Mat display8U;
    displayMat.convertTo(display8U, CV_8U);

    // Convert to QImage and scale to label
    //QImage pixmap = VizUtils::matToQImage(holoMat);
    QImage pixmap = VizUtils::matToQImage(display8U);
    //pixmap = pixmap.scaled(ui->HoloImgViz->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    pixmap = pixmap.scaled(ui->HoloImgViz->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->HoloImgViz->setPixmap(QPixmap::fromImage(pixmap));

    // Always update phase image (respects XYProj internally)
    updatePhaseImageDisplay();
}


// resizeEvent to dynamically adjust the image size
void RIHVR::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    updateHoloImageDisplay(); // also handles phase image if XYProj is selected
    if (!ui->XYProj->isChecked()) {
        updatePhaseImageDisplay(); // only update independently if XYProj not selected
    }
}

// helper function to load and show the phase image projection
void RIHVR::updatePhaseImageDisplay()
{
    QString imgPath = ui->PhaseVizImgPath->text().trimmed();
    if (imgPath.isEmpty())
        return;

    QImage image = VizUtils::loadImageAsQImage(imgPath);
    if (image.isNull()) {
        ui->PhaseImgViz->setText("Failed to load image");
        ui->PhaseImgViz->setAlignment(Qt::AlignCenter);
        return;
    }

    QPixmap pixmap = QPixmap::fromImage(image);

    // if XYProj selected, match size of HoloImg
    if (ui->XYProj->isChecked()) {
        pixmap = pixmap.scaled(ui->HoloImgViz->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        //pixmap = pixmap.scaled(ui->HoloImgViz->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    } else {
        //pixmap = pixmap.scaled(ui->PhaseImgViz->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        pixmap = pixmap.scaled(ui->PhaseImgViz->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    ui->PhaseImgViz->setPixmap(pixmap);
}


// load CSV tracks with frame information
std::vector<std::vector<std::pair<int, QVector3D>>> loadTracksFromCSV(const QString& csvPath)
{
    QFile file(csvPath);
    std::vector<std::vector<std::pair<int, QVector3D>>> tracks;

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "Error: Cannot open CSV file:" << csvPath;
        return tracks;
    }

    QTextStream in(&file);
    QString header = in.readLine().trimmed(); // read header line
    qDebug() << "CSV Header:" << header;

    // Map particle ID -> list of (frame, position)
    std::map<int, std::vector<std::pair<int, QVector3D>>> trackMap;

    int lineCount = 0;
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty()) continue;

        QStringList parts = line.split(QRegularExpression("[,;\\t]"), Qt::SkipEmptyParts);
        if (parts.size() < 5) continue; // x,y,z,frame,id

        bool ok = true;
        float x = parts[0].toFloat(&ok); if (!ok) continue;
        float y = parts[1].toFloat(&ok); if (!ok) continue;
        float z = parts[2].toFloat(&ok); if (!ok) continue;
        int frame = parts[3].toInt(&ok); if (!ok) continue;
        int id = parts[4].toInt(&ok); if (!ok) continue;

        trackMap[id].push_back({ frame, QVector3D(x, y, z) });
        ++lineCount;
    }

    file.close();
    qDebug() << "Read" << lineCount << "points across" << trackMap.size() << "particle tracks.";

    // Sort each track by frame and push into output vector
    for (auto& kv : trackMap) {
        auto& vec = kv.second;
        std::sort(vec.begin(), vec.end(),
                  [](const auto& a, const auto& b){ return a.first < b.first; });

        tracks.push_back(std::move(vec));  // now we store both frame and position
    }

    qDebug() << "Tracks organized. Total tracks:" << tracks.size();
    return tracks;
}


// --- Initialize OpenGL widget ---
void RIHVR::initializeTracksPlot()
{
    if (tracksPlotter) return; // already initialized

    // Create layout for the container
    QVBoxLayout* layout = new QVBoxLayout(ui->TracksViz);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    // Create the OpenGL track plot widget
    tracksPlotter = new TracksPlot(ui->TracksViz);
    layout->addWidget(tracksPlotter);

    // Set layout on the container
    ui->TracksViz->setLayout(layout);
}

// --- Slot to load tracks from text field ---
void RIHVR::loadTracksFromTextField()
{
    if (!tracksPlotter) {
        qWarning() << "TracksPlot widget not initialized!";
        return;
    }

    QString csvPath = ui->TracksVizPath->text().trimmed();
    if (csvPath.isEmpty()) {
        qWarning() << "TracksPath field is empty!";
        return;
    }

    // Load CSV once
    allTracks = loadTracksFromCSV(csvPath);  // now stores frame numbers
    if (allTracks.empty()) {
        qWarning() << "No tracks loaded from CSV!";
        return;
    }

    qDebug() << "Loaded" << allTracks.size() << "total tracks from CSV.";

    // total number of time steps
    maxFrameInTracks = 0;
    for (const auto& track : allTracks)
        if (!track.empty())
            maxFrameInTracks = std::max(maxFrameInTracks, track.back().first);

    // Apply initial filtering based on both sliders
    int concentrationValue = ui->TrackConcentration->value();  // 0–100
    int timeValue = ui->TimeSlider->value();                     // 0–maxFrame (or 0–100% mapped to frames)

    // Update tracks with both concentration and time applied
    updateTrackDisplay(concentrationValue, timeValue);

    qDebug() << "Tracks loaded and displayed from:" << csvPath;
}


// if the sliders changes, update the plot
void RIHVR::on_TrackConcentration_valueChanged(int value)
{
    if (allTracks.empty()) return;
    int maxFrame = ui->TimeSlider->value();
    updateTrackDisplay(value, maxFrame);
}

void RIHVR::on_TimeSlider_valueChanged(int value)
{
    if (allTracks.empty()) return;

    int frameToShow = static_cast<int>(maxFrameInTracks * (value / 100.0f));
    int concentration = ui->TrackConcentration->value();
    updateTrackDisplay(concentration, frameToShow);
}


// filtering the tracks based on concentration and time slider
void RIHVR::updateTrackDisplay(int concentrationPercentage, int maxFrame)
{
    if (allTracks.empty() || !tracksPlotter)
        return;

    concentrationPercentage = std::clamp(concentrationPercentage, 0, 100);
    int numToShow = static_cast<int>(allTracks.size() * (concentrationPercentage / 100.0f));
    if (numToShow < 1 && !allTracks.empty())
        numToShow = 1; // always show at least one track

    std::vector<std::vector<QVector3D>> filteredTracks;
    filteredTracks.reserve(numToShow);

    for (int i = 0; i < numToShow; ++i)
    {
        const auto& track = allTracks[i];
        std::vector<QVector3D> trackUpToFrame;
        for (const auto& [frame, pos] : track)
        {
            if (frame <= maxFrame)
                trackUpToFrame.push_back(pos);
        }
        if (!trackUpToFrame.empty())
            filteredTracks.push_back(std::move(trackUpToFrame));
    }

    tracksPlotter->setTracks(filteredTracks);
    tracksPlotter->update();

    // print out the number of tracks being displayed in the console
    // Build message
    int numFiltered = filteredTracks.size();
    int numAll = allTracks.size();
    double conc = concentrationPercentage;
    int frame = maxFrame;
    QString msg = QString("Showing %1 tracks of %2 (%3%) up to frame %4")
                      .arg(numFiltered).arg(numAll).arg(conc).arg(frame);

    // Log to debug console
    qDebug() << msg;

    // Get the current text
    QString currentText = ui->Console->toPlainText();

    // Find the last newline
    int lastNewline = currentText.lastIndexOf('\n');

    // Replace the last line instead of appending a new one
    if (lastNewline != -1) {
        currentText = currentText.left(lastNewline + 1) + msg;
    } else {
        currentText = msg;
    }

    // Update the console
    ui->Console->setPlainText(currentText);

    // (Optional) scroll to bottom
    ui->Console->verticalScrollBar()->setValue(ui->Console->verticalScrollBar()->maximum());

}

// ================================================================== Batch Automation page =======================================================================//
void RIHVR::on_ProcessAll_clicked() {
    ui->Console->append("<span style='color: cyan;'><b>Starting full computation pipeline...</b></span>");
    ui->Console->append("<span style='color: white;'></span>");

    // Start the first enabled step
    if (ui->ComputeBGCheckAuto->isChecked()) {
        m_pipelineRunning = true;
        ui->Console->append("<span style='color: cyan;'> Computing Background...</span>");
        ui->Console->append("<span style='color: white;'></span>");
        // Call the same function the BG button uses
        if (ui->MovingBG->isChecked()) {
            computeMovingBackground();
        } else if (ui->CCBG->isChecked()){
            computeCCBackground();
        } else {
            computeBackground();
        }
    }
    else if (ui->ProcessHoloCheckAuto->isChecked()) {
        m_pipelineRunning = true;
        ui->Console->append("<span style='color: cyan;'> Computing Holograms...</span>");
        ui->Console->append("<span style='color: white;'></span>");
        computeHolographicReconstruction();
    }
    else if (ui->ComputeTracksCheckAuto->isChecked()) {
        m_pipelineRunning = true;
        ui->Console->append("<span style='color: cyan;'> Computing Tracks...</span>");
        ui->Console->append("<span style='color: white;'></span>");
        computeTracks();
    }
    else {
        ui->Console->append("<span style='color: orange;'>No compute stages selected. Nothing to run.</span>");
        ui->Console->append("<span style='color: white;'></span>");
    }
    // swtich to the console tab
    ui->tabWidget->setCurrentIndex(4);
}

// function to do multi-rihvr file processing
void RIHVR::on_ProcessAllRIHVRFiles_clicked()
{

    QString dirPath = ui->RIHVRFilesDir->text().trimmed();
    if (dirPath.isEmpty()) {
        QMessageBox::warning(this, "Batch Processing", "Please specify a folder containing .rihvr files.");
        return;
    }

    QDir dir(dirPath);
    m_batchRIHVRFiles = dir.entryList(QStringList() << "*.rihvr", QDir::Files);

    if (m_batchRIHVRFiles.isEmpty()) {
        QMessageBox::information(this, "Batch Processing", "No .rihvr files found in the folder.");
        return;
    }

    // Print total files in console
    ui->Console->append(QString("<span style='color: cyan;'><b>Found %1 .rihvr files in folder.</b></span>")
                            .arg(m_batchRIHVRFiles.size()));
    ui->Console->append("<span style='color: WHITE;'> </span>");

    // reset batch state
    m_currentBatchIndex = 0;
    m_stopBatch = false;  // very important: allow batch to run -- otherwise it would not run
    m_batchRunning = true;  // start batch

    // start processing the first file
    processNextRIHVRFile();

    // swtich to the console tab
    ui->tabWidget->setCurrentIndex(4);
}

void RIHVR::processNextRIHVRFile()
{
    // check stop flag
    if (m_stopBatch) {
        ui->Console->append("<span style='color: red;'><b>Batch stopped by user.</b></span>");
        ui->Console->append("<span style='color: WHITE;'> </span>");
        return;
    }

    // check if we are done
    if (m_currentBatchIndex >= m_batchRIHVRFiles.size()) {
        ui->Console->append("<span style='color: green;'><b>Batch processing complete!</b></span>");
        ui->Console->append("<span style='color: WHITE;'> </span>");
        m_batchRunning = false;  // mark batch finished
        return;
    }

    // get current file
    QString fileName = m_batchRIHVRFiles[m_currentBatchIndex];
    QString fullPath = QDir(ui->RIHVRFilesDir->text()).absoluteFilePath(fileName);

    // switch to console
    ui->tabWidget->setCurrentIndex(4);

    // log and load file
    // Print which file is being loaded
    ui->Console->append(QString("<span style='color: cyan;'><b>Loading file %1/%2: %3</b></span>")
                            .arg(m_currentBatchIndex + 1)
                            .arg(m_batchRIHVRFiles.size())
                            .arg(fileName));
    ui->Console->append("<span style='color: WHITE;'> </span>");

    loadStateFromFile(fullPath);

    // start pipeline
    ui->Console->append(QString("<span style='color: yellow;'>Processing: %1</span>").arg(fileName));
    ui->Console->append("<span style='color: WHITE;'> </span>");
    on_ProcessAll_clicked();  // triggers asynchronous BG → Holo → Tracks

    // process events to keep GUI responsive
    QCoreApplication::processEvents();
}
