#ifndef TRACKSPLOT_H
#define TRACKSPLOT_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QVector3D>
#include <vector>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QPainter>


class TracksPlot : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core
{
    Q_OBJECT

public:
    explicit TracksPlot(QWidget *parent = nullptr);
    ~TracksPlot();

    void setTracks(const std::vector<std::vector<QVector3D>>& tracks);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
private:
    // Plasma colormap for velocity visualization (0..1 input)
    QVector3D plasmaColormap(float t) const;
    QVector3D m_domainMax;  // store the bounding box max corner (rounded)
    struct TrackGL {
        QOpenGLBuffer vbo;
        int vertexCount;
    };

    std::vector<TrackGL> m_trackBuffers;

    QOpenGLBuffer m_bboxVBO;
    QOpenGLBuffer m_axesVBO;

    QOpenGLShaderProgram* m_program;
    QMatrix4x4 m_proj, m_view;

    QVector3D m_center;
    float m_initialZoom;
    float m_zoom;
    float m_rotX, m_rotY;
    float m_panX, m_panY;
    float m_cameraBaseDist = 50.0f;

    QPoint m_lastMousePos;

    QVector3D m_bboxMin, m_bboxMax;

    std::vector<QVector3D> m_bboxVertices;
    std::vector<QVector3D> m_axesVertices;

    void updateTrackBuffers(const std::vector<std::vector<QVector3D>>& tracks);
    void updateBoundingBox();
    void updateAxes();
};

#endif // TRACKSPLOT_H
