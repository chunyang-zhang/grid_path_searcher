#include <iostream>
#include <ros/ros.h>
#include <ros/console.h>
#include <Eigen/Eigen>
#include "backward.hpp"

#define inf 1>>20
struct GridNode;
typedef GridNode* GridNodePtr;

///Search and prune neighbors for JPS 3D
struct JPS3DNeib {
	// for each (dx,dy,dz) these contain:
	//    ns: neighbors that are always added
	//    f1: forced neighbors to check
	//    f2: neighbors to add if f1 is forced
	int ns[27][3][26];
	int f1[27][3][12];
	int f2[27][3][12];
	// nsz contains the number of neighbors for the four different types of moves:
	// no move (norm 0):        26 neighbors always added
	//                          0 forced neighbors to check (never happens)
	//                          0 neighbors to add if forced (never happens)
	// straight (norm 1):       1 neighbor always added
	//                          8 forced neighbors to check
	//                          8 neighbors to add if forced
	// diagonal (norm sqrt(2)): 3 neighbors always added
	//                          8 forced neighbors to check
	//                          12 neighbors to add if forced
	// diagonal (norm sqrt(3)): 7 neighbors always added
	//                          6 forced neighbors to check
	//                          12 neighbors to add if forced
	static constexpr int nsz[4][2] = {{26, 0}, {1, 8}, {3, 12}, {7, 12}};
	JPS3DNeib();
	private:
	void Neib(int dx, int dy, int dz, int norm1, int dev, int& tx, int& ty, int& tz);
	void FNeib( int dx, int dy, int dz, int norm1, int dev,
	    int& fx, int& fy, int& fz,
	    int& nx, int& ny, int& nz);
};

struct GridNode
{     
    int id;        // 1--> open set, -1 --> closed set
    Eigen::Vector3d coord; 
    Eigen::Vector3i dir;   // direction of expanding
    Eigen::Vector3i index;
	
    bool is_path;
    double gScore, fScore;
    GridNodePtr cameFrom;
    std::multimap<double, GridNodePtr>::iterator nodeMapIt;
    uint8_t * occupancy; 

    GridNode(Eigen::Vector3i _index, Eigen::Vector3d _coord){  
		id = 0;
		is_path = false;
		index = _index;
		coord = _coord;
		dir   = Eigen::Vector3i::Zero();

		gScore = inf;
		fScore = inf;
		cameFrom = NULL;
    }

    GridNode(){};
    ~GridNode(){};
};

class gridPathFinder
{
	private:
		double getDiagHeu(GridNodePtr node1, GridNodePtr node2);
		double getManhHeu(GridNodePtr node1, GridNodePtr node2);
		double getEuclHeu(GridNodePtr node1, GridNodePtr node2);
		double getHeu(GridNodePtr node1, GridNodePtr node2);

		std::vector<GridNodePtr> retrievePath(GridNodePtr current);

		double resolution, inv_resolution;
		double tie_breaker = 1.0 + 1.0 / 10000;

		std::vector<GridNodePtr> expandedNodes;
		std::vector<GridNodePtr> gridPath;
		std::vector<GridNodePtr> endPtrList;
		
		int GLX_SIZE, GLY_SIZE, GLZ_SIZE;
		int GLXYZ_SIZE, GLYZ_SIZE;
		double gl_xl, gl_yl, gl_zl;
		double gl_xu, gl_yu, gl_zu;
		
		int tmp_id_x, tmp_id_y, tmp_id_z;
		int goal_x, goal_y, goal_z;

		uint8_t * data;

		GridNodePtr *** GridNodeMap;
		std::multimap<double, GridNodePtr> openSet;

		JPS3DNeib * jn3d;

	public:
		gridPathFinder( ){				
    		jn3d = new JPS3DNeib();
		};

		gridPathFinder(){};
		~gridPathFinder(){
			delete jn3d;
		};

		void setObs(const double coord_x, const double coord_y, const double coord_z);

		void initGridMap(double _resolution, Eigen::Vector3d global_xyz_l, Eigen::Vector3d global_xyz_u);
		void graphSearch(Eigen::Vector3d start_pt, Eigen::Vector3d end_pt, bool use_jps = false);

		inline void getJpsSucc(GridNodePtr currentPtr, std::vector<GridNodePtr> & neighborPtrSets, std::vector<double> & edgeCostSets);
		inline void getSucc   (GridNodePtr currentPtr, std::vector<GridNodePtr> & neighborPtrSets, std::vector<double> & edgeCostSets);
		inline bool hasForced(int x, int y, int z, int dx, int dy, int dz);
		bool jump(int x, int y, int z, int dx, int dy, int dz, int& new_x, int& new_y, int& new_z);
		inline bool isOccupied(int idx_x, int idx_y, int idx_z) const;
		inline bool isFree(int idx_x, int idx_y, int idx_z) const;

		void resetGrid(GridNodePtr ptr);
		void resetUsedGrids();

		std::vector<Eigen::Vector3d> getPath();
		std::vector<Eigen::Vector3d> getVisitedNodes();
		std::vector<Eigen::Vector3d> getCloseNodes();

		inline Eigen::Vector3d gridIndex2coord(const Eigen::Vector3i index) const
		{
		    Eigen::Vector3d pt;

		    pt(0) = ((double)index(0) + 0.5) * resolution + gl_xl;
		    pt(1) = ((double)index(1) + 0.5) * resolution + gl_yl;
		    pt(2) = ((double)index(2) + 0.5) * resolution + gl_zl;

		    return pt;
		};

		inline Eigen::Vector3i coord2gridIndex(const Eigen::Vector3d pt) const
		{
		    Eigen::Vector3i idx;
		    idx <<  std::min( std::max( int( (pt(0) - gl_xl) * inv_resolution), 0), GLX_SIZE - 1),
		            std::min( std::max( int( (pt(1) - gl_yl) * inv_resolution), 0), GLY_SIZE - 1),
		            std::min( std::max( int( (pt(2) - gl_zl) * inv_resolution), 0), GLZ_SIZE - 1);      		    
		  
		    return idx;
		};

		inline uint8_t IndexQuery( const Eigen::Vector3i index) const
		{   
			if (index(0) >= 0 && index(1) >= 0 && index(2) >= 0 && index(0) < GLX_SIZE && index(1) < GLY_SIZE && index(2) < GLZ_SIZE)
			    return data[index(0) * GLYZ_SIZE + index(1) * GLZ_SIZE + index(2)];
			else
				return 1.0;
		};

		inline uint8_t IndexQueryFast( const Eigen::Vector3i & index) const
		{   
			return data[index(0) * GLYZ_SIZE + index(1) * GLZ_SIZE + index(2)];
		};

		inline uint8_t IndexQuery( const int & index_x, const int & index_y, const int & index_z) const
		{      
		    return data[index_x * GLYZ_SIZE + index_y * GLZ_SIZE + index_z];
		};

		inline uint8_t CoordQuery(const Eigen::Vector3d & coord) const
		{   
		    Eigen::Vector3i index = coord2gridIndex(coord);

		    if (index(0) >= 0 && index(1) >= 0 && index(2) >= 0 && index(0) < GLX_SIZE && index(1) < GLY_SIZE && index(2) < GLZ_SIZE)
		    	return data[index(0) * GLYZ_SIZE + index(1) * GLZ_SIZE + index(2)];
			else
				return 1.0;
		};

		inline uint8_t CoordQuery( const double & pt_x, const double & pt_y, const double & pt_z ) const
		{   
		    Eigen::Vector3d coord(pt_x, pt_y, pt_z);
		    Eigen::Vector3i index = coord2gridIndex(coord);

		    if (index(0) >= 0 && index(1) >= 0 && index(2) >= 0 && index(0) < GLX_SIZE && index(1) < GLY_SIZE && index(2) < GLZ_SIZE)
		    	return data[index(0) * GLYZ_SIZE + index(1) * GLZ_SIZE + index(2)];
			else
				return 1.0;
		};
};