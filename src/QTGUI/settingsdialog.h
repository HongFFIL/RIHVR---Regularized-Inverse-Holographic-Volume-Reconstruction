#ifndef SETTINGSDIALOG_H
#define SETTINGSDIALOG_H

#include <QDialog>
#include <QString>

QT_BEGIN_NAMESPACE
namespace Ui { class SettingsDialog; }
QT_END_NAMESPACE

class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SettingsDialog(QWidget *parent = nullptr);
    ~SettingsDialog();

    // Getters for the paths
    QString getDefaultJsonPath() const;
    QString getProcessExePath() const;
    QString getSegmentationExePath() const;
    QString getCCBGExePath() const;
    QString getBackgroundExePath() const;
    QString getTrackingExePath() const;
    QString getOpenCVPath() const;
    QString getCudaPath() const;


    // setters for the paths
    void setDefaultJsonPath(const QString &path);
    void setProcessExePath(const QString &path);
    void setSegmentationExePath(const QString &path);
    void setCCBGExePath(const QString &path);
    void setBackgroundExePath(const QString &path);
    void setTrackingExePath(const QString &path);
    void setOpenCVPath(const QString &path);
    void setCudaPath(const QString &path);


private slots:
    void on_DefaultSettingsButton_clicked();
    void on_ProcessExeButton_clicked();
    void on_SegmentationExeButton_clicked();
    void on_CCBGExeButton_clicked();
    void on_BGExeButton_clicked();
    void on_TrackingExeButton_clicked();
    void on_SaveSettingsButton_clicked();
    void on_CancelSettingsButton_clicked();
    void on_OpenCVButton_clicked();
    void on_CudaButton_clicked();

private:
    Ui::SettingsDialog *ui;

    // Internal storage for paths
    QString defaultJsonPath;
    QString processExePath;
    QString segmentationExePath;
    QString backgroundExePath;
    QString trackingExePath;
    QString ccBGExePath;
    QString openCVPath;
    QString cudaPath;

    void loadSettings();
    void saveSettings();

    void accept() override;  // <-- must match exactly
};

#endif // SETTINGSDIALOG_H
