#include "blob3d.h"  // class implemented

using namespace umnholo;

bool BoundingBox::intersect(BoundingBox bb2)
{
    // Boxes intersect if either one bound of first is within both bounds
    // of second or first straddles the second
    bool xin = (((xmin <= bb2.xmax) && (xmin >= bb2.xmin)) ||
                ((xmax <= bb2.xmax) && (xmax >= bb2.xmin)) ||
                ((xmin <= bb2.xmin) && (xmax >= bb2.xmax)));
    bool yin = (((ymin <= bb2.ymax) && (ymin >= bb2.ymin)) ||
                ((ymax <= bb2.ymax) && (ymax >= bb2.ymin)) ||
                ((ymin <= bb2.ymin) && (ymax >= bb2.ymax)));
    bool zin = (((zmin <= bb2.zmax) && (zmin >= bb2.zmin)) ||
                ((zmax <= bb2.zmax) && (zmax >= bb2.zmin)) ||
                ((zmin <= bb2.zmin) && (zmax >= bb2.zmax)));
    
    return (xin && yin && zin);
}

void BoundingBox::expand(int dist)
{
    xmin -= dist;
    xmax += dist;
    ymin -= dist;
    ymax += dist;
    zmin -= dist;
    zmax += dist;
}

/////////////////////////////// PUBLIC ///////////////////////////////////////

//============================= LIFECYCLE ====================================

Blob3d::Blob3d()
{
    num_voxels = 0;
    source_width = 0;
    source_height = 0;
    centroid.x = 0;
    centroid.y = 0;
    centroid.z = 0;
    centroid_updated = false;
    
    global_offset.x = 0;
    global_offset.y = 0;
    global_offset.z = 0;
    
    bounds.is_set = false;
}

//============================= OPERATORS ====================================
//============================= OPERATIONS ===================================

void Blob3d::addVoxel(size_t global_idx)
{
    voxels.push_back(global_idx);
    num_voxels++;
    centroid_updated = false;
    return;
}

void Blob3d::addVoxel(cv::Point3i point)
{
    cv::Rect max_size(0,0,point.x, point.y);
    if ((point.x > source_width) || (point.y > source_height))
    {
        this->rebase(max_size);
    }
    
    size_t global_idx = point.z*source_width*source_height + 
                        point.y*source_width + 
                        point.x;
    this->addVoxel(global_idx);
    return;
}

void Blob3d::findBounds()
{
    if (num_voxels < 1) return;
    
    // Initialize bounds to first voxel, allows min and max to proceed
    cv::Point3i vox = getVoxel(0);
    bounds.xmin = vox.x;
    bounds.xmax = vox.x;
    bounds.ymin = vox.y;
    bounds.ymax = vox.y;
    bounds.zmin = vox.z;
    bounds.zmax = vox.z;
    
    for (int n = 1; n < num_voxels; ++n)
    {
        vox = getVoxel(n);
        bounds.xmin = std::min(vox.x, bounds.xmin);
        bounds.xmax = std::max(vox.x, bounds.xmax);
        bounds.ymin = std::min(vox.y, bounds.ymin);
        bounds.ymax = std::max(vox.y, bounds.ymax);
        bounds.zmin = std::min(vox.z, bounds.zmin);
        bounds.zmax = std::max(vox.z, bounds.zmax);
    }
    
    bounds.is_set = true;
    return;
}

void Blob3d::mergeIn(Blob3d obj2)
{
    for (int n = 0; n < obj2.num_voxels; ++n)
    {
        std::vector<size_t>::iterator it;
        it = std::find(voxels.begin(), voxels.end(), obj2.voxels[n]);
        if (it == voxels.end()) // did not find voxel
        {
            this->addVoxel(obj2.voxels[n]);
        }
    }
}

void Blob3d::rebase(cv::Rect new_source)
{
    size_t new_width = new_source.width;
    size_t new_height = new_source.height;
    
    if ((new_width > 0) && (new_height > 0))
    {
        for (int n = 0; n < this->num_voxels; ++n)
        {
            cv::Point3i pt = this->getVoxelGlobal(n);
            size_t idx = pt.z*new_width*new_height + pt.y*new_width + pt.x;
            this->voxels[n] = idx;
        }
        
        this->global_offset.x = new_source.x;
        this->global_offset.y = new_source.y;
        this->global_offset.z = 0;
    }
}

//============================= ACCESS     ===================================

void Blob3d::setSourceSize(size_t src_width, size_t src_height)
{
    source_width = src_width;
    source_height = src_height;
    
    return;
}

cv::Point3i Blob3d::getVoxelLocal(size_t id)
{
    if ((id < 0) || (id >= num_voxels))
    {
        cv::Point3i pt(0,0,0);
        return pt;
    }
    
    if ((source_width == 0) || (source_height == 0))
    {
        std::cout << "Blob3d::getVoxel Error: " << std::endl;
        std::cout << "Source width and height must be specified to find centroid" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    size_t plane_size = source_width * source_height;
    size_t vox = voxels[id];
    cv::Point3i pt;
    pt.z = vox / plane_size;
    pt.y = (vox - plane_size*pt.z) / source_width;
    pt.x = vox - (plane_size*pt.z) - pt.y*source_width;
    
    return pt;
}

cv::Point3i Blob3d::getVoxelGlobal(size_t id)
{
    cv::Point3i pt = getVoxelLocal(id);
    
    pt.x += global_offset.x;
    pt.y += global_offset.y;
    pt.z += global_offset.z;
    
    return pt;
}

void Blob3d::setValue(size_t id, float value)
{
    assert(id < num_voxels);
    
    if (values.size() < num_voxels)
    {
        values.resize(num_voxels);
    }
    
    values[id] = value;
}

cv::Point3f Blob3d::getCentroid()
{
    if (centroid_updated) return centroid;
    
    if ((source_width == 0) || (source_height == 0))
    {
        std::cout << "Blob3d::getCentroid Error: " << std::endl;
        std::cout << "Source width and height must be specified to find centroid" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    cv::Point3i sums(0,0,0);
    size_t plane_size = source_width * source_height;
    for (int n = 0; n < num_voxels; ++n)
    {
        cv::Point3i vox = getVoxel(n);
        
        sums.x += vox.x;
        sums.y += vox.y;
        sums.z += vox.z;
    }
    
    centroid.x = (float)sums.x / (float)num_voxels;
    centroid.y = (float)sums.y / (float)num_voxels;
    centroid.z = (float)sums.z / (float)num_voxels;
    
    centroid_updated = true;
    return centroid;
}

cv::Point3f Blob3d::getWeightedCentroid()
{
    if (weighted_centroid_updated) return weighted_centroid;
    
    if ((source_width == 0) || (source_height == 0))
    {
        std::cout << "Blob3d::getWeightedCentroid Error: " << std::endl;
        std::cout << "Source width and height must be specified to find centroid" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    if (values.size() != num_voxels)
    {
        std::cout << "Warning: selected getWeightedCentroid but no weights set\n";
        return getCentroid();
    }
    
    cv::Point3f sums(0,0,0);
    float weight_sums = 0;
    size_t plane_size = source_width * source_height;
    for (int n = 0; n < num_voxels; ++n)
    {
        cv::Point3i vox = getVoxel(n);
        float weight = values[n];
        
        sums.x += weight * vox.x;
        sums.y += weight * vox.y;
        sums.z += weight * vox.z;
        weight_sums += weight;
    }
    
    weighted_centroid.x = (float)sums.x / weight_sums;
    weighted_centroid.y = (float)sums.y / weight_sums;
    weighted_centroid.z = (float)sums.z / weight_sums;
    
    weighted_centroid_updated = true;
    return weighted_centroid;
}

float Blob3d::getFocusZ()
{
    if (!bounds.is_set)
        findBounds();
    
    return (bounds.zmin + bounds.zmax) / 2.0;
}

CuMat Blob3d::getFocusMask()
{
    if ((source_width == 0) || (source_height == 0))
    {
        std::cout << "Blob3d::getFocusMask Error: source size unspecified" << std::endl;
        throw HOLO_ERROR_INVALID_STATE;
    }
    
    CuMat mask;
    cv::Mat mask_data = cv::Mat::zeros(source_height, source_width, CV_8U);
    
    int focus_z = getFocusZ();
    
    for (int n = 0; n < num_voxels; ++n)
    {
        cv::Point3i vox = getVoxel(n);
        //if (vox.z == focus_z)
        {
            // Add 1 to each index to match matlab. Not sure why though
            mask_data.at<uint8_t>(vox.y+1, vox.x+1) = 1;
        }
    }
    
    mask.setMatData(mask_data);
    return mask;
}

int Blob3d::getLength()
{
    if (!bounds.is_set)
        findBounds();
    
    return bounds.zmax - bounds.zmin;
}

BoundingBox Blob3d::getBounds()
{
    if (!bounds.is_set)
    {
        this->findBounds();
    }
    BoundingBox b = bounds;
    return b;
}

float Blob3d::getMaxValue()
{
    float maxval = *std::max_element(values.begin(), values.end());
    return maxval;
}

float Blob3d::getMeanValue()
{
    float sumval = std::accumulate(values.begin(), values.end(), 0.0);
    return sumval / (float)num_voxels;
}

//============================= INQUIRY    ===================================

bool Blob3d::intersects(Blob3d obj2)
{
    // First compare bounding boxes
    this->findBounds();
    obj2.findBounds();
    BoundingBox bb1 = this->bounds;
    BoundingBox bb2 = obj2.bounds;
    
    if (bb1.intersect(bb2))
    {
        // If objects are close enough, test all voxels for inclusion
        for (int n = 0; n < obj2.num_voxels; ++n)
        {
            std::vector<size_t>::iterator it;
            it = std::find(voxels.begin(), voxels.end(), obj2.voxels[n]);
            if (it != voxels.end()) // voxel[n] present in this object
                return true;
        }
    }
    
    return false;
}

bool Blob3d::overlaps(Blob3d obj2, float dist)
{
    // First compare bounding boxes
    this->findBounds();
    obj2.findBounds();
    BoundingBox bb1 = this->bounds;
    bb1.expand(dist);
    BoundingBox bb2 = obj2.bounds;
    
    if (bb1.intersect(bb2))
    {
        // First, test for inclusion without distance buffer
        for (int n = 0; n < obj2.num_voxels; ++n)
        {
            std::vector<size_t>::iterator it;
            it = std::find(voxels.begin(), voxels.end(), obj2.voxels[n]);
            if (it != voxels.end()) // voxel[n] present in this object
                return true;
        }
        
        // Now naively compare each voxel to each other voxel
        for (int n2 = 0; n2 < obj2.num_voxels; ++n2)
        {
            cv::Point3i vox2 = obj2.getVoxel(n2);
            for (int n1 = 0; n1 < this->num_voxels; ++n1)
            {
                cv::Point3i vox1 = this->getVoxel(n1);
                float dx = (float)(vox1.x - vox2.x);
                float dy = (float)(vox1.y - vox2.y);
                float dz = (float)(vox1.z - vox2.z);
                float d = sqrt(dx*dx + dy*dy + dz*dz);
                
                if (d < dist)
                {
                    return true;
                }
            }
        }
    }
    
    return false;
}

/////////////////////////////// PROTECTED  ///////////////////////////////////

/////////////////////////////// PRIVATE    ///////////////////////////////////
