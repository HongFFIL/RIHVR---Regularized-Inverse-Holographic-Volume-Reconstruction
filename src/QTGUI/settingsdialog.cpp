#include "settingsdialog.h"
#include "ui_settingsdialog.h"
#include <QFileDialog>
#include <QSettings>
#include <QMessageBox>

// for save button
#include <QStandardPaths>
#include <QDir>
#include <QFile>
#include <QTextStream>

SettingsDialog::SettingsDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SettingsDialog)
{
    ui->setupUi(this);
    loadSettings();


    // Connect buttons if you prefer explicit connect
    //connect(ui->DefaultSettingsButton, &QPushButton::clicked, this, &SettingsDialog::on_DefaultSettingsButton_clicked);
    //connect(ui->ProcessExeButton, &QPushButton::clicked, this, &SettingsDialog::on_ProcessExeButton_clicked);
    //connect(ui->SegmentationExeButton, &QPushButton::clicked, this, &SettingsDialog::on_SegmentationExeButton_clicked);
    //connect(ui->BGExeButton, &QPushButton::clicked, this, &SettingsDialog::on_BGExeButton_clicked);
    //connect(ui->SaveSettingsButton, &QPushButton::clicked, this, &SettingsDialog::on_SaveSettingsButton_clicked);
}

SettingsDialog::~SettingsDialog()
{
    delete ui;
}

// the cancel button in the settings dialog box
void SettingsDialog::on_CancelSettingsButton_clicked()
{
    this->close();   // or simply 'close();'
}

// Load saved paths from QSettings
void SettingsDialog::loadSettings()
{
    QSettings settings("FFIL", "RIHVR");

    defaultJsonPath     = settings.value("defaultJsonPath", "").toString();
    processExePath      = settings.value("processExePath", "").toString();
    segmentationExePath = settings.value("segmentationExePath", "").toString();
    backgroundExePath   = settings.value("backgroundExePath", "").toString();
    trackingExePath   = settings.value("trackingExePath", "").toString();
    ccBGExePath   = settings.value("ccBGExePath", "").toString();
    openCVPath   = settings.value("OpenCVPath", "").toString();
    cudaPath   = settings.value("CudaPath", "").toString();

    ui->DefaultSettingsPath->setText(defaultJsonPath);
    ui->ProcessExePath->setText(processExePath);
    ui->SegmentationExePath->setText(segmentationExePath);
    ui->BGExePath->setText(backgroundExePath);
    ui->CCBGExePath->setText(ccBGExePath);
    ui->TrackingExePath->setText(trackingExePath);
    ui->OpenCVPath->setText(openCVPath);
    ui->CudaPath->setText(cudaPath);
}

void SettingsDialog::setDefaultJsonPath(const QString &path) { ui->DefaultSettingsPath->setText(path); }
void SettingsDialog::setProcessExePath(const QString &path) { ui->ProcessExePath->setText(path); }
void SettingsDialog::setSegmentationExePath(const QString &path) { ui->SegmentationExePath->setText(path); }
void SettingsDialog::setBackgroundExePath(const QString &path) { ui->BGExePath->setText(path); }
void SettingsDialog::setTrackingExePath(const QString &path) { ui->TrackingExePath->setText(path); }
void SettingsDialog::setCCBGExePath(const QString &path) { ui->CCBGExePath->setText(path); }
void SettingsDialog::setOpenCVPath(const QString &path) { ui->OpenCVPath->setText(path); }
void SettingsDialog::setCudaPath(const QString &path) { ui->CudaPath->setText(path); }

// Save current paths to QSettings
void SettingsDialog::saveSettings()
{
    QSettings settings("FFIL", "RIHVR");
    settings.setValue("defaultJsonPath", defaultJsonPath);
    settings.setValue("processExePath", processExePath);
    settings.setValue("segmentationExePath", segmentationExePath);
    settings.setValue("backgroundExePath", backgroundExePath);
    settings.setValue("ccBGExePath", ccBGExePath);
    settings.setValue("trackingBGExePath", trackingExePath);
    settings.setValue("OpenCVPath", openCVPath);
    settings.setValue("CudaPath", cudaPath);

    QMessageBox::information(this, "Saved", "Settings saved successfully!");
}

QString SettingsDialog::getDefaultJsonPath() const { return defaultJsonPath; }
QString SettingsDialog::getProcessExePath() const { return processExePath; }
QString SettingsDialog::getSegmentationExePath() const { return segmentationExePath; }
QString SettingsDialog::getBackgroundExePath() const { return backgroundExePath; }
QString SettingsDialog::getCCBGExePath() const { return ccBGExePath; }
QString SettingsDialog::getTrackingExePath() const { return trackingExePath; }
QString SettingsDialog::getOpenCVPath() const { return openCVPath; }
QString SettingsDialog::getCudaPath() const { return cudaPath; }

// Slot implementations
void SettingsDialog::on_DefaultSettingsButton_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, tr("Select Default .rihvr File"), "", tr("RIHVR Files (*.rihvr)"));
    if (!file.isEmpty()) {
        defaultJsonPath = file;
        ui->DefaultSettingsPath->setText(file);
    }
}

void SettingsDialog::on_ProcessExeButton_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, tr("Select sparse-inverse-recon.exe"), "", tr("Executable Files (*)"));
    if (!file.isEmpty()) {
        processExePath = file;
        ui->ProcessExePath->setText(file);
    }
}

void SettingsDialog::on_SegmentationExeButton_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, tr("Select segmentation.exe"), "", tr("Executable Files (*)"));
    if (!file.isEmpty()) {
        segmentationExePath = file;
        ui->SegmentationExePath->setText(file);
    }
}

void SettingsDialog::on_TrackingExeButton_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, tr("Select particletracking.exe"), "", tr("Executable Files (*)"));
    if (!file.isEmpty()) {
        trackingExePath = file;
        ui->TrackingExePath->setText(file);
    }
}

void SettingsDialog::on_BGExeButton_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, tr("Select make-background.exe"), "", tr("Executable Files (*)"));
    if (!file.isEmpty()) {
        backgroundExePath = file;
        ui->BGExePath->setText(file);
    }
}

void SettingsDialog::on_CCBGExeButton_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, tr("Select ccbg.exe"), "", tr("Executable Files (*)"));
    if (!file.isEmpty()) {
        ccBGExePath = file;
        ui->CCBGExePath->setText(file);
    }
}

void SettingsDialog::on_CudaButton_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this,
                                                    tr("Select CUDA bin folder containing the dlls"),
                                                    ui->CudaPath->text(),
                                                    QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
        cudaPath = dir;
        ui->CudaPath->setText(dir);
}

void SettingsDialog::on_OpenCVButton_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this,
                                                    tr("Select OpenCV bin folder containing the dlls"),
                                                    ui->OpenCVPath->text(),
                                                    QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    openCVPath = dir;
    ui->OpenCVPath->setText(dir);
}



void SettingsDialog::on_SaveSettingsButton_clicked()
{
    // Get the text from the line edit
    QString defaultPath = ui->DefaultSettingsPath->text().trimmed();

    if (defaultPath.isEmpty()) {
        QMessageBox::warning(this, "Missing Path", "Please select or enter a default settings path first.");
        return;
    }

    // Build the full path to the startup file
    QString dirPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)
                      + "/RIHVR_FFIL";

    // Ensure the directory exists
    QDir().mkpath(dirPath);

    QString startupPathFile = dirPath + "/startuppath.txt";

    // Try writing to the file
    QFile file(startupPathFile);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Error", "Failed to write to " + startupPathFile);
        return;
    }

    QTextStream out(&file);
    out << defaultPath << Qt::endl;
    file.close();

    QMessageBox::information(this, "Success", "Default settings path saved successfully!");

    // Close the dialog cleanly
    this->accept();  // dialog returns QDialog::Accepted
}

void SettingsDialog::accept()
{
    // Update member variables from UI
    defaultJsonPath     = ui->DefaultSettingsPath->text().trimmed();
    processExePath      = ui->ProcessExePath->text().trimmed();
    segmentationExePath = ui->SegmentationExePath->text().trimmed();
    backgroundExePath   = ui->BGExePath->text().trimmed();
    trackingExePath     = ui->TrackingExePath->text().trimmed();
    ccBGExePath         = ui->CCBGExePath->text().trimmed();
    openCVPath          = ui->OpenCVPath->text().trimmed();
    cudaPath            = ui->CudaPath->text().trimmed();

    // Save to QSettings
    QSettings settings("FFIL", "RIHVR");
    settings.setValue("defaultJsonPath", defaultJsonPath);
    settings.setValue("processExePath", processExePath);
    settings.setValue("segmentationExePath", segmentationExePath);
    settings.setValue("backgroundExePath", backgroundExePath);
    settings.setValue("trackingExePath", trackingExePath);
    settings.setValue("ccBGExePath", ccBGExePath);
    settings.setValue("OpenCVPath", openCVPath);
    settings.setValue("CudaPath", cudaPath);

    // save defaultPath to startup file
    QString dirPath = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation)
                      + "/RIHVR_FFIL";
    QDir().mkpath(dirPath);
    QFile file(dirPath + "/startuppath.txt");
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        out << defaultJsonPath << Qt::endl;
        file.close();
    }

    QDialog::accept();  // now close the dialog
}

