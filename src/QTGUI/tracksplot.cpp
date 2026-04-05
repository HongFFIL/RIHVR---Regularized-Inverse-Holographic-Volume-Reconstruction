#include "tracksplot.h"
#include <QOpenGLShader>
#include <algorithm>
#include <cmath> // for std::ceil, std::pow

TracksPlot::TracksPlot(QWidget *parent)
    : QOpenGLWidget(parent),
    m_program(nullptr),
    m_rotX(20.0f), m_rotY(-30.0f),
    m_zoom(1.0f), m_initialZoom(1.0f),
    m_panX(0.0f), m_panY(0.0f)
{
}

TracksPlot::~TracksPlot()
{
    makeCurrent();
    for (auto& t : m_trackBuffers) t.vbo.destroy();
    m_bboxVBO.destroy();
    m_axesVBO.destroy();
    delete m_program;
    doneCurrent();
}

void TracksPlot::setTracks(const std::vector<std::vector<QVector3D>>& tracks)
{
    makeCurrent();
    updateTrackBuffers(tracks);
    updateBoundingBox();
    updateAxes();
    doneCurrent();
    update();
}

void TracksPlot::initializeGL()
{
    initializeOpenGLFunctions();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(2.0f);

    // Shader program
    m_program = new QOpenGLShaderProgram(this);
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                       "#version 330 core\n"
                                       "layout(location = 0) in vec3 position;\n"
                                       "layout(location = 1) in vec3 color;\n"
                                       "out vec3 fragColor;\n"
                                       "uniform mat4 mvp;\n"
                                       "void main() { fragColor = color; gl_Position = mvp * vec4(position,1.0); }"
                                       );
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                       "#version 330 core\n"
                                       "in vec3 fragColor;\n"
                                       "out vec4 outColor;\n"
                                       "void main() { outColor = vec4(fragColor,1.0); }"
                                       );
    m_program->link();
}

void TracksPlot::resizeGL(int w, int h)
{
    float aspect = float(w) / float(h ? h : 1);
    m_proj.setToIdentity();

    // Bounding box diagonal
    float diag = (m_bboxMax - m_bboxMin).length();
    float cameraDist = m_cameraBaseDist * m_zoom;

    // Dynamic near/far to ensure nothing disappears
    float nearPlane = 0.01f;                    // very close near plane
    float farPlane  = cameraDist + diag * 5.0f; // far enough for entire scene

    m_proj.perspective(45.0f, aspect, nearPlane, farPlane);
}

void TracksPlot::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_view.setToIdentity();

    // --- camera transform ---
    // camera always looks at particle center
    m_view.translate(-m_center + QVector3D(-m_panX, -m_panY, 0.0f));
    m_view.translate(0, 0, -m_cameraBaseDist * m_zoom);
    m_view.rotate(m_rotX, 1, 0, 0);
    m_view.rotate(m_rotY, 0, 1, 0);

    // Recompute projection dynamically to prevent disappearing
    float diag = (m_bboxMax - m_bboxMin).length();
    float cameraDist = m_cameraBaseDist * m_zoom;
    QMatrix4x4 proj;
    proj.setToIdentity();
    float aspect = float(width()) / float(height() ? height() : 1);
    proj.perspective(45.0f, aspect, 0.01f, cameraDist + diag * 5.0f);

    QMatrix4x4 mvp = proj * m_view;

    m_program->bind();
    m_program->setUniformValue("mvp", mvp);

    // Draw tracks (yellow)
    for (auto& track : m_trackBuffers) {
        track.vbo.bind();
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(QVector3D)*2, nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(QVector3D)*2, reinterpret_cast<void*>(sizeof(QVector3D)));
        glDrawArrays(GL_LINE_STRIP, 0, track.vertexCount);
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        track.vbo.release();
    }

    // Draw bounding box (white)
    if (m_bboxVertices.size() > 0) {
        glLineWidth(1.0f);  // thin bounding box
        m_bboxVBO.bind();
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(QVector3D)*2, nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(QVector3D)*2, reinterpret_cast<void*>(sizeof(QVector3D)));
        glDrawArrays(GL_LINES, 0, int(m_bboxVertices.size()));
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        m_bboxVBO.release();
    }

    // Draw axes (thicker, after bounding box)
    if (m_axesVertices.size() > 0) {
        glLineWidth(5.0f); // thicker axes
        m_axesVBO.bind();
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(QVector3D)*2, nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(QVector3D)*2, reinterpret_cast<void*>(sizeof(QVector3D)));
        glDrawArrays(GL_LINES, 0, int(m_axesVertices.size()));
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        m_axesVBO.release();
    }

    m_program->release();

    // --- Draw bounding box labels using QPainter overlay ---
    QPainter painter(this);
    painter.setPen(Qt::white);
    painter.setFont(QFont("Arial", 14, QFont::Bold));

    auto projectToScreen = [&](const QVector3D &pt) -> QPointF {
        QVector4D clip = m_proj * m_view * QVector4D(pt,1.0f);
        if (clip.w() != 0.0f) clip /= clip.w();
        float x = (clip.x() * 0.5f + 0.5f) * width();
        float y = (1.0f - (clip.y() * 0.5f + 0.5f)) * height();
        return QPointF(x,y);
    };

    // Bounding box labels
    QVector3D origin(0,0,0);
    QVector3D farCorner = m_domainMax; // use domain max instead of particle max

    // Points along axes
    QVector3D xTip(farCorner.x(), 0, 0);
    QVector3D yTip(0, farCorner.y(), 0);
    QVector3D zTip(0, 0, farCorner.z());

    painter.drawText(projectToScreen(origin), "0");
    painter.drawText(projectToScreen(xTip), "X");
    painter.drawText(projectToScreen(yTip), "Y");
    painter.drawText(projectToScreen(zTip), "Z");
    painter.drawText(projectToScreen(farCorner),
                     QString("(%1,%2,%3)").arg(farCorner.x()).arg(farCorner.y()).arg(farCorner.z()));

    painter.end();

}

void TracksPlot::mousePressEvent(QMouseEvent *event)
{
    m_lastMousePos = event->pos();
}

void TracksPlot::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - m_lastMousePos.x();
    int dy = event->y() - m_lastMousePos.y();

    if (event->buttons() & Qt::LeftButton) {
        // Rotate scene
        m_rotX += dy;
        m_rotY += dx;
    }
    else if (event->buttons() & Qt::RightButton) {
        // Intuitive pan (scene moves with mouse)
        // Factor based on scene size
        float sceneScale = (m_bboxMax - m_bboxMin).length();
        float factor = sceneScale * 0.002f;   // tweak for speed preference

        m_panX -= dx * factor;  // drag right → scene moves right
        m_panY += dy * factor;  // drag up → scene moves up
    }

    m_lastMousePos = event->pos();
    update();
}

void TracksPlot::wheelEvent(QWheelEvent *event)
{
    float delta = event->angleDelta().y() / 1200.0f;
    m_zoom *= (1.0f - delta);
    //m_zoom = qBound(0.01f, m_zoom, 10.0f);
    m_zoom = qBound(0.001f, m_zoom, 100.0f);   // larger zoom range
    update();
}

void TracksPlot::updateTrackBuffers(const std::vector<std::vector<QVector3D>>& tracks)
{
    for (auto& t : m_trackBuffers) t.vbo.destroy();
    m_trackBuffers.clear();
    if (tracks.empty()) return;

    QVector3D minPt( std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max());
    QVector3D maxPt = -minPt;

    // Compute global bbox
    for (const auto& track : tracks) {
        for (const auto& p : track) {
            minPt.setX(std::min(minPt.x(), p.x()));
            minPt.setY(std::min(minPt.y(), p.y()));
            minPt.setZ(std::min(minPt.z(), p.z()));
            maxPt.setX(std::max(maxPt.x(), p.x()));
            maxPt.setY(std::max(maxPt.y(), p.y()));
            maxPt.setZ(std::max(maxPt.z(), p.z()));
        }
    }

    m_bboxMin = minPt;
    m_bboxMax = maxPt;
    m_center = (minPt + maxPt) * 0.5f;
    float diag = (maxPt - minPt).length();
    m_cameraBaseDist = diag * 1.5f;
    m_zoom = 1.0f;

    // --- Precompute plasma LUT ---
    const int LUT_SIZE = 256;
    QVector<QVector3D> plasmaLUT(LUT_SIZE);
    for (int i = 0; i < LUT_SIZE; ++i) {
        float t = float(i)/(LUT_SIZE-1);
        plasmaLUT[i] = plasmaColormap(t);
    }

    auto lookupColor = [&](float t) -> QVector3D {
        t = std::clamp(t, 0.0f, 1.0f);
        float idxf = t * (LUT_SIZE-1);
        int idx0 = int(idxf);
        int idx1 = std::min(idx0+1, LUT_SIZE-1);
        float f = idxf - idx0;
        return plasmaLUT[idx0]*(1-f) + plasmaLUT[idx1]*f;
    };

    // --- Compute global vmax ---
    float globalVmax = 0.0f;
    for (const auto& track : tracks) {
        if (track.size() < 2) continue;
        for (size_t i = 1; i < track.size(); ++i) {
            float v = (track[i] - track[i-1]).length();
            globalVmax = std::max(globalVmax, v);
        }
    }
    if (globalVmax <= 0) globalVmax = 1.0f; // avoid divide by zero

    // --- Create VBOs with smooth velocity coloring ---
    for (const auto& track : tracks) {
        if (track.size() < 2) continue;
        TrackGL tgl;
        tgl.vertexCount = track.size();
        std::vector<QVector3D> interleaved;

        // Compute per-segment velocities and smooth
        std::vector<float> speeds(track.size(), 0.0f);
        for (size_t i = 1; i < track.size(); ++i)
            speeds[i] = (track[i] - track[i-1]).length();

        std::vector<float> smoothSpeeds(track.size(), 0.0f);
        smoothSpeeds[0] = speeds[0];
        smoothSpeeds[track.size()-1] = speeds.back();
        for (size_t i = 1; i+1 < track.size(); ++i)
            smoothSpeeds[i] = (speeds[i-1] + speeds[i] + speeds[i+1]) / 3.0f;

        // Interleave vertices + colors (normalized by globalVmax)
        for (size_t i = 0; i < track.size(); ++i) {
            QVector3D color = lookupColor(smoothSpeeds[i]/globalVmax);
            interleaved.push_back(track[i]);
            interleaved.push_back(color);
        }

        tgl.vbo.create();
        tgl.vbo.bind();
        tgl.vbo.allocate(interleaved.data(), int(interleaved.size()*sizeof(QVector3D)));
        tgl.vbo.release();

        m_trackBuffers.push_back(std::move(tgl));
    }
}

void TracksPlot::updateBoundingBox()
{
    std::vector<QVector3D> corners;

    QVector3D domainMin(0,0,0); // keep origin as min

    // Compute a margin (10% of the largest extent)
    QVector3D extent = m_bboxMax - domainMin;
    float maxExtent = std::max({extent.x(), extent.y(), extent.z()});
    float margin = maxExtent * 0.05f; // 5% margin

    QVector3D rawMax = m_bboxMax + QVector3D(margin, margin, margin);

    // Round each axis to nearest “nice” number
    auto roundNice = [](float val) -> float {
        if (val == 0) return 0;
        // Round to nearest multiple of 10, 50, 100, or 200 depending on magnitude
        float magnitude = std::pow(10.0f, std::floor(std::log10(val))); // base power of 10
        float factor = 1.0f;
        if (magnitude >= 1000) factor = 200.0f;
        else if (magnitude >= 500) factor = 100.0f;
        else if (magnitude >= 100) factor = 50.0f;
        else if (magnitude >= 10) factor = 10.0f;
        else factor = 1.0f;
        return std::ceil(val / factor) * factor;
    };


    QVector3D domainMax(
        roundNice(rawMax.x()),
        roundNice(rawMax.y()),
        roundNice(rawMax.z())
        );

    // Store domainMax for potential labels or scaling
    m_domainMax = domainMax;

    // 8 corners + connecting edges (like your existing code)
    corners = {
               {domainMin.x(), domainMin.y(), domainMin.z()}, {domainMax.x(), domainMin.y(), domainMin.z()},
               {domainMax.x(), domainMin.y(), domainMin.z()}, {domainMax.x(), domainMax.y(), domainMin.z()},
               {domainMax.x(), domainMax.y(), domainMin.z()}, {domainMin.x(), domainMax.y(), domainMin.z()},
               {domainMin.x(), domainMax.y(), domainMin.z()}, {domainMin.x(), domainMin.y(), domainMin.z()},

               {domainMin.x(), domainMin.y(), domainMax.z()}, {domainMax.x(), domainMin.y(), domainMax.z()},
               {domainMax.x(), domainMin.y(), domainMax.z()}, {domainMax.x(), domainMax.y(), domainMax.z()},
               {domainMax.x(), domainMax.y(), domainMax.z()}, {domainMin.x(), domainMax.y(), domainMax.z()},
               {domainMin.x(), domainMax.y(), domainMax.z()}, {domainMin.x(), domainMin.y(), domainMax.z()},

               {domainMin.x(), domainMin.y(), domainMin.z()}, {domainMin.x(), domainMin.y(), domainMax.z()},
               {domainMax.x(), domainMin.y(), domainMin.z()}, {domainMax.x(), domainMin.y(), domainMax.z()},
               {domainMax.x(), domainMax.y(), domainMin.z()}, {domainMax.x(), domainMax.y(), domainMax.z()},
               {domainMin.x(), domainMax.y(), domainMin.z()}, {domainMin.x(), domainMax.y(), domainMax.z()},
               };

    // Interleave color = white
    m_bboxVertices.clear();
    for (auto& c : corners) {
        m_bboxVertices.push_back(c);
        m_bboxVertices.push_back(QVector3D(1,1,1));
    }

    if (!m_bboxVBO.isCreated()) m_bboxVBO.create();
    m_bboxVBO.bind();
    m_bboxVBO.allocate(m_bboxVertices.data(), int(m_bboxVertices.size() * sizeof(QVector3D)));
    m_bboxVBO.release();
}

void TracksPlot::updateAxes()
{
    std::vector<QVector3D> axes;
    float len = (m_bboxMax - m_bboxMin).length() * 0.5f;

    QVector3D origin(0,0,0); // true world origin
    // X axis = red
    axes.push_back(origin); axes.push_back(QVector3D(1,0,0));
    axes.push_back(origin + QVector3D(len,0,0)); axes.push_back(QVector3D(1,0,0));
    // Y axis = green
    axes.push_back(origin); axes.push_back(QVector3D(0,1,0));
    axes.push_back(origin + QVector3D(0,len,0)); axes.push_back(QVector3D(0,1,0));
    // Z axis = blue
    axes.push_back(origin); axes.push_back(QVector3D(0,0,1));
    axes.push_back(origin + QVector3D(0,0,len)); axes.push_back(QVector3D(0,0,1));

    m_axesVertices = axes;

    if (!m_axesVBO.isCreated()) m_axesVBO.create();
    m_axesVBO.bind();
    m_axesVBO.allocate(m_axesVertices.data(), int(m_axesVertices.size() * sizeof(QVector3D)));
    m_axesVBO.release();
}

// Plasma colormap approximation (0..1 normalized input)
QVector3D TracksPlot::plasmaColormap(float t) const {
    t = std::clamp(t, 0.0f, 1.0f);
    // Plasma approximation (or lookup)
    float r = std::min(std::max(0.0f, 0.050383f + 0.625f*t - 1.0f*t*t + 0.5f*t*t*t), 1.0f);
    float g = std::min(std::max(0.0f, 0.029937f + 0.32f*t + 0.1f*t*t - 0.15f*t*t*t), 1.0f);
    float b = std::min(std::max(0.0f, 0.527f - 1.2f*t + 1.1f*t*t - 0.3f*t*t*t), 1.0f);
    return QVector3D(r, g, b);
}

