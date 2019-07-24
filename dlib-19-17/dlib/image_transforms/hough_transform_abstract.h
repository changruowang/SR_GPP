// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_HOUGH_tRANSFORM_ABSTRACT_Hh_
#ifdef DLIB_HOUGH_tRANSFORM_ABSTRACT_Hh_

#include "../geometry.h"
#include "../image_processing/generic_image.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class hough_transform
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a tool for computing the line finding version of the Hough
                transform given some kind of edge detection image as input.  It also allows
                the edge pixels to be weighted such that higher weighted edge pixels
                contribute correspondingly more to the output of the Hough transform,
                allowing stronger edges to create correspondingly stronger line detections
                in the final Hough transform.

            THREAD SAFETY
                It is safe for multiple threads to make concurrent accesses to this object
                without synchronization.
        !*/

    public:

        explicit hough_transform (
            unsigned long size_
        ); 
        /*!
            requires
                - size_ > 0
            ensures
                - This object will compute Hough transforms that are size_ by size_ pixels.  
                  This is in terms of both the Hough accumulator array size as well as the
                  input image size.
                - #size() == size_
        !*/

        unsigned long size(
        ) const;
        /*!
            ensures
                - returns the size of the Hough transforms generated by this object.  In
                  particular, this object creates Hough transform images that are size() by
                  size() pixels in size.
        !*/

        long nr(
        ) const;
        /*!
            ensures
                - returns size()
        !*/

        long nc(
        ) const;
        /*!
            ensures
                - returns size()
        !*/

        std::pair<dpoint, dpoint> get_line (
            const dpoint& p
        ) const;
        /*!
            requires
                - rectangle(0,0,size()-1,size()-1).contains(p) == true
                  (i.e. p must be a point inside the Hough accumulator array)
            ensures
                - returns the line segment in the original image space corresponding
                  to Hough transform point p. 
                - The returned points are inside rectangle(0,0,size()-1,size()-1).
        !*/

        double get_line_angle_in_degrees (
            const dpoint& p 
        ) const;
        /*!
            requires
                - rectangle(0,0,size()-1,size()-1).contains(p) == true
                  (i.e. p must be a point inside the Hough accumulator array)
            ensures
                - returns the angle, in degrees, of the line corresponding to the Hough
                  transform point p.
        !*/

        void get_line_properties (
            const dpoint& p,
            double& angle_in_degrees,
            double& radius
        ) const;
        /*!
            requires
                - rectangle(0,0,size()-1,size()-1).contains(p) == true
                  (i.e. p must be a point inside the Hough accumulator array)
            ensures
                - Converts a point in the Hough transform space into an angle, in degrees,
                  and a radius, measured in pixels from the center of the input image.
                - #angle_in_degrees == the angle of the line corresponding to the Hough
                  transform point p.  Moreover: -90 <= #angle_in_degrees < 90.
                - #radius == the distance from the center of the input image, measured in
                  pixels, and the line corresponding to the Hough transform point p.
                  Moreover: -sqrt(size()*size()/2) <= #radius <= sqrt(size()*size()/2)
                - Note that the line properties are calculated to sub-pixel accuracy.  That
                  is, p doesn't have to contain integer values, it can reference locations
                  between pixels and the appropriate calculation will be done to find the
                  corresponding line.
        !*/

        template <
            typename image_type
            >
        point get_best_hough_point (
            const point& p,
            const image_type& himg
        );
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h and it must contain grayscale pixels.
                - himg.nr() == size()
                - himg.nc() == size()
                - rectangle(0,0,size()-1,size()-1).contains(p) == true
            ensures
                - This function interprets himg as a Hough image and p as a point in the
                  original image space.  Given this, it finds the maximum scoring line that
                  passes though p.  That is, it checks all the Hough accumulator bins in
                  himg corresponding to lines though p and returns the location with the
                  largest score.  
                - returns a point X such that get_rect(himg).contains(X) == true
        !*/

        template <
            typename in_image_type,
            typename out_image_type
            >
        void operator() (
            const in_image_type& img,
            const rectangle& box,
            out_image_type& himg
        ) const;
        /*!
            requires
                - in_image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h and it must contain grayscale pixels.
                - out_image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h and it must contain grayscale pixels.
                - box.width() == size()
                - box.height() == size()
            ensures
                - Computes the Hough transform of the part of img contained within box.
                  In particular, we do a grayscale version of the Hough transform where any
                  non-zero pixel in img is treated as a potential component of a line and
                  accumulated into the Hough accumulator #himg.  However, rather than
                  adding 1 to each relevant accumulator bin we add the value of the pixel
                  in img to each Hough accumulator bin.  This means that, if all the
                  pixels in img are 0 or 1 then this routine performs a normal Hough
                  transform.  However, if some pixels have larger values then they will be
                  weighted correspondingly more in the resulting Hough transform.
                - #himg.nr() == size()
                - #himg.nc() == size()
                - #himg is the Hough transform of the part of img contained in box.  Each
                  point in #himg corresponds to a line in the input box.  In particular,
                  the line for #himg[y][x] is given by get_line(point(x,y)).  Also, when
                  viewing the #himg image, the x-axis gives the angle of the line and the
                  y-axis the distance of the line from the center of the box.  The
                  conversion between Hough coordinates and angle and pixel distance can be
                  obtained by calling get_line_properties().
        !*/

        template <
            typename in_image_type,
            typename out_image_type
            >
        void operator() (
            const in_image_type& img,
            out_image_type& himg
        ) const;
        /*!
            requires
                - in_image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h and it must contain grayscale pixels.
                - out_image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h and it must contain grayscale pixels.
                - num_rows(img) == size()
                - num_columns(img) == size()
            ensures
                - performs: (*this)(img, get_rect(img), himg);
                  That is, just runs the hough transform on the whole input image.
        !*/

        template <
            typename in_image_type
            >
        std::vector<std::vector<point>> find_pixels_voting_for_lines (
            const in_image_type& img,
            const rectangle& box,
            const std::vector<point>& hough_points,
            const unsigned long angle_window_size = 1,
            const unsigned long radius_window_size = 1
        ) const;
        /*!
            requires
                - in_image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h and it must contain grayscale pixels.
                - box.width() == size()
                - box.height() == size()
                - for all valid i:
                    - get_rect(*this).contains(hough_points[i]) == true
                      (i.e. hough_points must contain points in the output Hough transform
                      space generated by this object.)
                - angle_window_size >= 1
                - radius_window_size >= 1
            ensures
                - This function computes the Hough transform of the part of img contained
                  within box.  It does the same computation as operator() defined above,
                  except instead of accumulating into an image we create an explicit list
                  of all the points in img that contributed to each line (i.e each point in
                  the Hough image). To do this we take a list of Hough points as input and
                  only record hits on these specifically identified Hough points.  A
                  typical use of find_pixels_voting_for_lines() is to first run the normal
                  Hough transform using operator(), then find the lines you are interested
                  in, and then call find_pixels_voting_for_lines() to determine which
                  pixels in the input image belong to those lines.
                - This routine returns a vector, CONSTITUENT_POINTS, with the following
                  properties:
                    - #CONSTITUENT_POINTS.size() == hough_points.size()
                    - for all valid i:
                        - Let HP[i] = centered_rect(hough_points[i], angle_window_size, radius_window_size)
                        - Any point in img with a non-zero value that lies on a line
                          corresponding to one of the Hough points in HP[i] is added to
                          CONSTITUENT_POINTS[i].  Therefore, when this routine finishes,
                          #CONSTITUENT_POINTS[i] will contain all the points in img that
                          voted for the lines associated with the Hough accumulator bins in
                          HP[i].
                        - #CONSTITUENT_POINTS[i].size() == the number of points in img that
                          voted for any of the lines HP[i] in Hough space.  Note, however,
                          that if angle_window_size or radius_window_size are made so large
                          that HP[i] overlaps HP[j] for i!=j then the overlapping regions
                          of Hough space are assigned to HP[i] or HP[j] arbitrarily.
                          That is, we treat HP[i] and HP[j] as disjoint even if their boxes
                          overlap.  In this case, the overlapping region is assigned to
                          either HP[i] or HP[j] in an arbitrary manner.
        !*/

        template <
            typename in_image_type
            >
        std::vector<std::vector<point>> find_pixels_voting_for_lines (
            const in_image_type& img,
            const std::vector<point>& hough_points,
            const unsigned long angle_window_size = 1,
            const unsigned long radius_window_size = 1
        ) const;
        /*!
            requires
                - in_image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h and it must contain grayscale pixels.
                - num_rows(img) == size()
                - num_columns(img) == size()
                - for all valid i:
                    - get_rect(*this).contains(hough_points[i]) == true
                      (i.e. hough_points must contain points in the output Hough transform
                      space generated by this object.)
                - angle_window_size >= 1
                - radius_window_size >= 1
            ensures
                - performs: return find_pixels_voting_for_lines(img, get_rect(img), hough_points, angle_window_size, radius_window_size);
                  That is, just runs the routine on the whole input image.
        !*/

        template <
            typename image_type,
            typename thresh_type
            >
        std::vector<point> find_strong_hough_points(
            const image_type& himg,
            const thresh_type hough_count_threshold,
            const double angle_nms_thresh,
            const double radius_nms_thresh
        );
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h and it must contain grayscale pixels.
                - himg.nr() == size()
                - himg.nc() == size()
                - angle_nms_thresh >= 0
                - radius_nms_thresh >= 0
            ensures
                - This routine finds strong lines in a Hough transform and performs
                  non-maximum suppression on the detected lines.  Recall that each point in
                  Hough space is associated with a line. Therefore, this routine finds all
                  the pixels in himg (a Hough transform image) with values >=
                  hough_count_threshold and performs non-maximum suppression on the
                  identified list of pixels.  It does this by discarding lines that are
                  within angle_nms_thresh degrees of a stronger line or within
                  radius_nms_thresh distance (in terms of radius as defined by
                  get_line_properties()) to a stronger Hough point.
                - The identified lines are returned as a list of coordinates in himg.
                - The returned points are sorted so that points with larger Hough transform
                  values come first.
        !*/

        template <
            typename in_image_type,
            typename record_hit_function_type
            >
        void perform_generic_hough_transform (
            const in_image_type& img,
            const rectangle& box,
            record_hit_function_type record_hit
        ) const;
        /*!
            requires
                - in_image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h and it must contain grayscale pixels.
                - box.width() == size()
                - box.height() == size()
                - record_hit is a function object with the signature:
                    void record_hit(const point& hough_point, const point& img_point, in_image_pixel_type value)
            ensures
                - Computes the Hough transform of the part of img contained within box.
                  This routine is very general and allows you to implement a wide variety
                  of Hough transforms, in fact, the operator() and
                  find_pixels_voting_for_lines() routines defined above are implemented in
                  terms of perform_generic_hough_transform().  The behavior is described by
                  the following pseudo-code:
                    for (image_coordinate : all_coordinates_in_img)
                        for (hough_point : all_Hough_space_coordinates_for_lines_passing_through_image_coordinate)
                            record_hit(hough_point, image_coordinate, img[image_coordinate.y][image_coordinate.x()]);
                  That is, we perform the Hough transform, but rather than accumulating
                  into a Hough accumulator image, we call record_hit() and record_hit()
                  does whatever it wants.  For example, in the operator() method defined
                  above record_hit() simply accumulates into an image, and therefor
                  performs the classic Hough transform.  But there are many other options.
        !*/

        template <
            typename in_image_type,
            typename record_hit_function_type
            >
        void perform_generic_hough_transform (
            const in_image_type& img,
            record_hit_function_type record_hit
        ) const;
        /*!
            requires
                - in_image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h and it must contain grayscale pixels.
                - record_hit is a function object with the signature:
                    void record_hit(const point& hough_point, const point& img_point, in_image_pixel_type value)
                - num_rows(img) == size()
                - num_columns(img) == size()
            ensures
                - performs: perform_generic_hough_transform(img, get_rect(img), record_hit);
                  That is, just runs the routine on the whole input image.
        !*/

    };
}

#endif // DLIB_HOUGH_tRANSFORM_ABSTRACT_Hh_


