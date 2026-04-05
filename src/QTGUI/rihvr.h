#ifndef RIHVR_H
#define RIHVR_H

#include "settingsdialog.h"
#include <QMainWindow>
#include <QProcess>
#include "tracksplot.h" // for TracksVisualizer

#include <QVector3D>
#include <QString>
#include <vector>

// Forward declaration of CSV loader
//std::vector<std::vector<QVector3D>> loadTracksFromCSV(const QString& csvPath);
std::vector<std::vector<std::pair<int, QVector3D>>> loadTracksFromCSV(const QString& csvPath);

QT_BEGIN_NAMESPACE
namespace Ui {
class RIHVR;
}
QT_END_NAMESPACE

class RIHVR : public QMainWindow
{
    Q_OBJECT

// Signals that are emitted when the process is finished
signals:
    void computeBGFinished(bool success);
    void computeTracksFinished(bool success);
    void computeHoloFinished(bool success);
    void processAllFinished(bool success);

public:
    RIHVR(QWidget *parent = nullptr);
    ~RIHVR();

private:
    Ui::RIHVR *ui;
private:
    int maxFrameInTracks = 0; // computed after loading CSV
    bool m_pipelineRunning = false; // to deal with running processes in a batch sequence

    void on_ProcessAll_clicked();
private:
    TracksPlot* tracksPlotter = nullptr; // persistent OpenGL widget

    void initializeTracksPlot();

private slots:
    void loadTracksFromTextField();

private slots:
    void on_TrackConcentration_valueChanged(int value);  // slot for slider
    void on_TimeSlider_valueChanged(int value);  // slot for slider
    void updateTrackDisplay(int concentrationPercentage, int maxFrame);             // helper to apply filter

private:
    std::vector<std::vector<std::pair<int, QVector3D>>> allTracks; // full set of tracks from CSV

// private slot for showing the about message box
private slots:
    void showAboutDialog();
    void showLicenseDialog();

// private slot for showing the load state and save state menu
private slots:
    void saveStateToFile();
    void loadStateFromFile(const QString &filePath = QString());    //(const QString &filePath = QString(), bool showMessage = true);

// private slot for enabling/disabling groups in the background tab based on compute bg options
private slots:
    void updateBackgroundTab();

// private slot for updating the HoloVizPath automatically
private slots:
    void updateHoloVizPath();
// private slot for updating the PhaseVizPath automatically
private slots:
    void updatePhaseVizPath();

// private slot for enabling/disabling the compute BG button
private:
    bool canEnableComputeBGButton();

// private slot for declaring the function that displays the holoimg
private:
    void updateHoloImageDisplay();
// private slot for declaring the function that displays the phase projection image
private:
    void updatePhaseImageDisplay();
// private slot for handling the image resizing
protected:
    void resizeEvent(QResizeEvent *event) override;

// declaration for the startup file loader
    void loadStartupPath();
// declaration for syncing the phase and holo sliders
    void setupSliderSync();
// browse button for holoimg path
    void on_HoloVizImgBrowseButton_clicked();

// a pointer to the settings dialog so that we can get the EXE paths
    SettingsDialog *settingsdialog;   // pointer to settings dialog
private:
    QString currentDefaultRihvrPath;  // stores the path even if the settings dialog is closed

// declare the median and mean background computation function
public slots:
    void computeBackground();

// declare the correlation-based background computation function
public slots:
    void computeCCBackground();

// declare the moving background computation function
public slots:
    void computeMovingBackground();
private slots:
    void processNextMovingBG(int bgWindowSize, int bgFirstImage, int bgEndImage,
                             int numImages, int startImage,
                             const QString& inputPath, const QString& inputFileFormat,
                             const QString& outputPath,
                             const QString& bgExePath, const QString& cudaPath,
                             const QString& openCVPath);
// to keep track of the moving frames
private:
    QList<int> m_movingBGFrames;

// to keep track of the holo frames
private:
    QList<int> m_HoloFrames;

// to keep track of the ongoing processes -- we will need them to use with the stop button
private:
    QList<QProcess*> m_activeProcesses;

// to track .rihvr files being processed
private:
    QStringList m_batchRIHVRFiles;  // list of .rihvr files in batch
    int m_currentBatchIndex = -1;   // index of current file being processed

// declare the stop button function
public slots:
    void stopAllProcesses();
// flag to check if a stop has been requested
private:
    bool m_stopRequested = false;
    bool m_stopBatch = false; // for stopping all the batch processes when doing multiple .rihvr files
    bool m_batchRunning = false; // flag to know if we are doing .rihvr batching or not

// declare the clear button function
public slots:
    void clearConsole();
public slots:
    void on_ProcessAllRIHVRFiles_clicked();
    void processNextRIHVRFile();
signals:
    void pipelineFinished(bool success); // to signal that the pipeline has finished so that we can move to the next .rihvr file

// declare the saveLog button console function
public slots:
    void saveLogConsole();

// define a variable to share the image path between imgpath and holoimg path
private:
    QString imgPathShared;  // shared between background and processing pages
// define a variable to share the bg path between imgpath and holoimg path
private:
    QString bgPathShared;  // shared between background and processing pages
// define a variable to share the image name file format between BG and processing pages
private:
    QString imgFileFormatShared;  // shared between background and processing pages
// define a variable to share the start and end image index between BG and processing pages
private:
    QString imgStartIdXShared;  // shared between background and processing pages
private:
    QString imgEndIdXShared;  // shared between background and processing pages

// decalre variables to share information between processing and tracking page
private:
    QString trackStartIdXShared;  // shared between tracking and processing pages
private:
    QString trackEndIdXShared;  // shared between tracking and processing pages
private:
    QString trackPathShared;  // shared between tracking and viz pages

// to keep track of the final progress of processing
private:
    bool AllDoneFlag = false;

// declare the holographic reconstruction computation functions
public slots:
    void computeHolographicReconstruction();
private slots:
    void processNextHologram(int startImg, int endImg, int increment,
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
                             const QString& cudaPath, const QString& openCVPath);

// declare the correlation-based background computation function
public slots:
    void computeTracks();


};


#endif // RIHVR_H

